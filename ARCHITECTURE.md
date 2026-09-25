# L3A Architecture Record

Team phải cập nhật tài liệu này cùng source. Mục tiêu là mô tả quyết định có thể kiểm chứng, không ghi prompt bí mật hoặc chain-of-thought.

## 1. System overview

```text
inputs/<case_id>.json
        │
        ▼
   Coordinator ── list_tools (MCP discovery)
        │
        ▼
  order_agent ──get_order──► item_agent ──get_order_items,get_sellers──►
  payment_agent ──get_order_payments,get_payment_timeline──►
  shipment_agent ──get_shipment_summary──►
  refund_agent ──get_refund_timeline──►
  policy_agent ──get_policy──► verifier_agent
        │                                        │
        └──────────── MCP Evidence Gateway ──────┴──── trace.jsonl (LangGraph StateGraph, linear pipeline)
        ▼
outputs/<case_id>.json
```

`solve_case` (`src/student_agent/workflow.py`) compiles a LangGraph `StateGraph` with one node per
specialist. Each node only receives the state slice it needs (order_id, previously fetched
domain data) and returns its own contribution; LangGraph merges these into shared `State`.

## 2. Agent ownership

| Actor | Input | Trách nhiệm | Output/handoff |
| --- | --- | --- | --- |
| Coordinator | case, discovered tool list | Discover tools, assign order lookup | `order_id` (from customer's claim, later corrected by evidence) |
| order_agent | `order_id` | Call `get_order`, resolve the authoritative `order_id`/`order_status` | `order_data` → item_agent |
| item_agent | `order_id` | Call `get_order_items` + `get_sellers` to resolve item/seller identities and item-level totals (the order row itself carries no total) | `item_data`, `seller_data` → payment_agent |
| payment_agent | `order_id`, `order_data` | Call `get_order_payments` + `get_payment_timeline`; compute paid total, detect duplicate authorizations | `payment_data`, `payment_timeline_data` → shipment_agent |
| shipment_agent | `order_id`, `order_data` | Call `get_shipment_summary`; carrier handoff / delivery timestamps | `shipment_data` → refund_agent |
| refund_agent | `order_id` | Call `get_refund_timeline`; determine whether a refund request is stuck pending or has failed | `refund_data` → policy_agent |
| policy_agent | all collected evidence | Call `get_policy`; run the rule-based classifier (`_build_conclusion`) to pick `primary_issue`, responsible party, and refund lines | draft `final_answer` → verifier_agent |
| verifier_agent | draft `final_answer` | Recompute `recommended_refund_brl` from `refund_lines`, clamp confidence to [0,1], attach only the evidence refs relevant to each claim's topic (`TOPIC_DOMAINS`) | final output |

Chỉ agent tương ứng gọi tool domain của mình; không agent nào gọi tool ngoài phạm vi nêu trên.

## 3. A2A protocol

Mỗi node nhận/trả một `State` (TypedDict) được LangGraph định tuyến tuần tự theo cạnh cố định
(`coordinator → order_agent → item_agent → payment_agent → shipment_agent → refund_agent →
policy_agent → verifier_agent`). Mọi state đều mang `case_id` implicit qua closure của
`solve_case`; không có message rời rạc giữa case khác nhau vì mỗi lời gọi `solve_case` tạo một
graph run độc lập. Handoff giữa agent chỉ trace `event_type=handoff` với `decision_code` phản ánh
việc evidence có thu được hay không (`*_evidence_collected` / `*_evidence_missing`) — không trace
nội dung suy luận. Không có vòng lặp: đồ thị là DAG tuyến tính, mỗi node chạy đúng một lần.

## 4. Evidence lifecycle

Mọi lời gọi tool đi qua `EvidenceGateway.call` (`src/student_agent/mcp_gateway.py`), luôn kèm
`case_id` và validate response theo `mcp-evidence-response-v1.schema.json` trước khi trả về. Mỗi
`evidence_ref` được lưu vào `state["evidence_refs"]` (để hard-gate provenance) và
`state["refs_by_domain"][domain]` (để verifier chọn evidence liên quan cho từng claim). Ngay khi
evidence được dùng để suy luận, `trace.emit(event_type="tool_result_consumed", ...)` ghi lại
`tool_name` + `evidence_refs`. Evidence không bao giờ được tái sử dụng giữa các case vì
`solve_case` không giữ state toàn cục — mỗi case có `evidence_refs`/`refs_by_domain` riêng khởi
tạo lại từ `coordinator`.

## 5. Failure policy

| Failure | Retry? | Fallback | Trace event/code |
| --- | --- | --- | --- |
| MCP tool call raises (`RuntimeError`/`ValueError`) | Không retry | `_fetch` trả `None`; node coi domain đó là thiếu evidence | `handoff` với `decision_code=*_evidence_missing` |
| Order not found (`order_data` rỗng) | Không retry | `_build_conclusion` trả `primary_issue=insufficient_evidence`, mọi claim → `verdict=insufficient_evidence` | `policy_decided` với `decision_code=insufficient_evidence` |
| Nguồn xung đột (order vs shipment delivered date lệch >1 ngày) | N/A | Ưu tiên `shipment` (tool mô tả rõ là nguồn timestamp giao hàng), ghi vào `data_conflicts` thay vì âm thầm chọn | field xuất hiện trong `data_conflicts[]` của output |
| Kết quả specialist không hợp lệ | N/A | `contracts.validate_output` chặn output sai schema trước khi ghi file (`cli.py`) | ngoại lệ `ContractError`, case không được ghi ra `outputs/` |

Không có retry vì lỗi tool được coi là "không có evidence", không phải lỗi tạm thời cần thử lại;
việc này tránh gọi tool trùng lặp không cần thiết trên cùng case. Missing evidence không bao giờ
được thay bằng dữ liệu suy đoán — các trường liên quan giữ giá trị rỗng/`None` và độ tin cậy giảm
tương ứng.

## 6. Verification invariants

`verifier_agent` kiểm tra trước khi finalize:

- **Schema**: `contracts.validate_output` (gọi từ `cli.py` sau khi `solve_case` trả về) chặn mọi
  output không khớp `l3a-output-v2.schema.json`.
- **Entity scope**: `affected_entities` chỉ chứa id trích từ evidence đã fetch của chính case đó
  (`order_id`, `item_ids`/`seller_ids` từ `get_order_items`/`get_sellers`, không có giá trị đoán).
- **Evidence ownership**: mỗi claim chỉ mang `evidence_refs` thuộc domain thực sự liên quan đến
  topic của nó (`TOPIC_DOMAINS`), không gán toàn bộ evidence cho mọi claim.
- **Claim linkage**: claim `insufficient_evidence` luôn có `evidence_refs=[]`; claim khớp
  `primary_issue` hoặc `requested_full_refund` được gán evidence phù hợp.
- **Money totals**: `recommended_refund_brl` được recompute lại từ tổng `refund_lines` trong
  verifier để đảm bảo hai trường luôn nhất quán, dù logic ở `policy_agent` có sai lệch nhỏ do rounding.
- **Responsibility/action consistency**: `responsible_parties` chỉ gán `party_id` khi đúng loại
  bên chịu trách nhiệm có id xác thực (vd. `seller` → seller_id thật; `logistics_provider` không
  có id khả dụng nên để `null` thay vì mượn seller_id).
- **Confidence bounds**: `assessment.confidence` được clamp về `[0, 1]` trước khi trả kết quả.

## 7. Reproducibility

- Runtime: Python ≥3.11 (dev venv hiện dùng 3.14), dependencies pin theo `pyproject.toml`
  (`langgraph`, `mcp`, `jsonschema[format]`, `python-dotenv`, `httpx2`).
- Không dùng LLM trong `solve_case`: toàn bộ phân loại `primary_issue` là rule-based, xác định
  (deterministic) dựa trên dữ liệu MCP trả về — cùng input luôn cho cùng output, không có seed
  ngẫu nhiên nào cần ghi lại.
- Concurrency: mỗi case chạy tuần tự trong `cli.py::_run`; các MCP call trong một case cũng tuần
  tự (không fan-out song song) để giữ trace đơn giản và tránh vượt call budget không cần thiết.
- Lệnh chạy: `day09 mcp-tools`, `day09 run`, `day09 validate`, `day09 package --output dist/submission.zip`.
- Không ghi API key hay giá trị `.env` vào repo hoặc log; `Settings.load` chỉ đọc từ biến môi trường/`.env` cục bộ.
