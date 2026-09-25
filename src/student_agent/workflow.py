from __future__ import annotations

from datetime import datetime
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .mcp_gateway import EvidenceGateway
from .trace import TraceWriter

ALL_DOMAINS = (
    "order",
    "item",
    "seller",
    "payment",
    "payment_timeline",
    "shipment",
    "refund",
    "policy",
)

# Which evidence domains genuinely support each claim topic / primary issue.
# Used so the verifier only cites evidence that was actually examined for
# that specific claim, instead of dumping every ref on every claim.
TOPIC_DOMAINS: dict[str, tuple[str, ...]] = {
    "canceled_order_paid": ("order", "payment"),
    "unavailable_order_paid": ("order", "payment", "item", "seller"),
    "late_delivery_seller": ("order", "shipment", "item", "policy"),
    "late_delivery_logistics": ("order", "shipment", "policy"),
    "valid_split_payment": ("order", "payment"),
    "payment_mismatch": ("order", "payment", "item"),
    "duplicate_charge": ("order", "payment", "payment_timeline"),
    "refund_pending": ("refund", "payment"),
    "refund_failed": ("refund", "payment"),
    "requested_full_refund": ("payment", "refund"),
    "unsupported_claim": ALL_DOMAINS,
    "insufficient_evidence": (),
}


class State(TypedDict, total=False):
    case: dict[str, Any]
    tools: list[str]

    order_id: str | None
    order_data: dict[str, Any]
    item_data: list[dict[str, Any]]
    seller_data: list[dict[str, Any]]
    payment_data: dict[str, Any]
    payment_timeline_data: dict[str, Any]
    shipment_data: dict[str, Any]
    refund_data: dict[str, Any]
    policy_data: dict[str, Any]

    evidence_refs: list[str]
    refs_by_domain: dict[str, str]
    final_answer: dict[str, Any]


def _pick(data: dict[str, Any] | None, *keys: str) -> Any:
    if not isinstance(data, dict):
        return None
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return value
    return None


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _date(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.strip().replace("Z", "+00:00").replace(" ", "T")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _as_list(data: Any, *keys: str) -> list[dict[str, Any]]:
    """Normalize an evidence payload into a list of row dicts.

    Tool responses may return a bare list, or a dict wrapping the rows under
    one of several plausible keys, or a single row dict.
    """
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        row_markers = ("product_id", "seller_id", "payment_value", "event_type", "status")
        if any(marker in data for marker in row_markers):
            return [data]
    return []


def _payment_total(payment_data: dict[str, Any] | None) -> float | None:
    if not isinstance(payment_data, dict):
        return None
    total = _num(_pick(payment_data, "payment_value", "total_paid", "amount"))
    if total is not None:
        return total
    entries = _as_list(payment_data, "payments", "installments")
    values = [_num(_pick(entry, "payment_value", "amount")) for entry in entries]
    values = [value for value in values if value is not None]
    if values:
        return sum(values)
    return None


def _items_summary(item_rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw_item_ids = {_pick(row, "product_id", "order_item_id", "item_id") for row in item_rows}
    item_ids = sorted(raw_item_ids - {None})
    seller_ids = sorted({_pick(row, "seller_id") for row in item_rows} - {None})
    amounts = [_num(_pick(row, "price")) or 0.0 for row in item_rows]
    freights = [_num(_pick(row, "freight_value", "freight")) or 0.0 for row in item_rows]
    items_total = round(sum(amounts) + sum(freights), 2) if item_rows else None
    limit_dates = [_date(_pick(row, "shipping_limit_date")) for row in item_rows]
    limit_dates = [d for d in limit_dates if d is not None]
    latest_limit = max(limit_dates) if limit_dates else None
    return {
        "item_ids": item_ids,
        "seller_ids": seller_ids,
        "items_total": items_total,
        "shipping_limit_at": latest_limit,
    }


def _latest_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not events:
        return None
    dated = [
        (event, _date(_pick(event, "occurred_at", "event_at", "timestamp", "updated_at")))
        for event in events
    ]
    if any(d for _, d in dated):
        dated.sort(key=lambda pair: pair[1] or datetime.min)
        return dated[-1][0]
    return events[-1]


def _refund_status(refund_events: list[dict[str, Any]]) -> str | None:
    if not refund_events:
        return None
    latest = _latest_event(refund_events)
    status = str(_pick(latest, "status", "event_type", "state") or "").lower()
    if status in {"failed", "rejected", "denied", "declined"}:
        return "failed"
    if status in {"pending", "processing", "requested", "initiated", "approved", "in_review"}:
        return "pending"
    if status in {"completed", "refunded", "paid", "settled"}:
        return "completed"
    return None


def _has_duplicate_authorization(timeline_events: list[dict[str, Any]]) -> bool:
    capture_statuses = {"authorized", "captured", "paid", "approved"}
    amounts: list[float] = []
    for event in timeline_events:
        status = str(_pick(event, "status", "event_type") or "").lower()
        if status in capture_statuses:
            amount = _num(_pick(event, "amount", "payment_value"))
            if amount is not None:
                amounts.append(round(amount, 2))
    return len(amounts) != len(set(amounts)) and len(amounts) > 1


def _resolve_tool(tools: list[str], *keywords: str) -> str | None:
    for keyword in keywords:
        exact = f"get_{keyword}"
        if exact in tools:
            return exact
    lowered = {tool: tool.lower() for tool in tools}
    for keyword in keywords:
        for tool, tool_lower in lowered.items():
            if keyword in tool_lower:
                return tool
    return None


async def solve_case(
    case: dict[str, Any], gateway: EvidenceGateway, trace: TraceWriter
) -> dict[str, Any]:
    case_id = case["case_id"]
    claimed_order_id = case.get("customer_request", {}).get("claimed_order_id")

    async def _fetch(
        state: State, tool_name: str | None, actor: str, domain: str, **arguments: Any
    ) -> Any:
        if tool_name is None:
            return None
        clean_args = {key: value for key, value in arguments.items() if value is not None}
        try:
            evidence = await gateway.call(tool_name, case_id=case_id, **clean_args)
        except (RuntimeError, ValueError):
            return None
        evidence_ref = evidence["evidence_ref"]
        data = evidence.get("data")
        refs = list(state.get("evidence_refs", []))
        if evidence_ref not in refs:
            refs.append(evidence_ref)
        state["evidence_refs"] = refs
        by_domain = dict(state.get("refs_by_domain", {}))
        by_domain[domain] = evidence_ref
        state["refs_by_domain"] = by_domain
        trace.emit(
            case_id=case_id,
            event_type="tool_result_consumed",
            actor=actor,
            tool_name=tool_name,
            evidence_refs=[evidence_ref],
        )
        return data

    async def coordinator(state: State) -> dict[str, Any]:
        tools = await gateway.list_tools()
        trace.emit(
            case_id=case_id,
            event_type="task_assigned",
            actor="coordinator",
            target="order_agent",
            decision_code="collect_order_evidence",
        )
        return {
            "tools": tools,
            "evidence_refs": [],
            "refs_by_domain": {},
            "order_id": claimed_order_id,
        }

    async def order_agent(state: State) -> dict[str, Any]:
        tool_name = _resolve_tool(state["tools"], "order")
        order_id = state.get("order_id")
        data = await _fetch(state, tool_name, "order-agent", "order", order_id=order_id)
        data = data or {}
        order_id = _pick(data, "order_id") or state.get("order_id")
        trace.emit(
            case_id=case_id,
            event_type="handoff",
            actor="order-agent",
            target="item_agent",
            decision_code="order_evidence_collected" if data else "order_evidence_missing",
        )
        return {
            "order_data": data,
            "order_id": order_id,
            "evidence_refs": state["evidence_refs"],
            "refs_by_domain": state["refs_by_domain"],
        }

    async def item_agent(state: State) -> dict[str, Any]:
        trace.emit(
            case_id=case_id,
            event_type="task_assigned",
            actor="coordinator",
            target="item_agent",
            decision_code="collect_item_evidence",
        )
        order_id = state.get("order_id")
        items_tool = _resolve_tool(state["tools"], "order_items")
        sellers_tool = _resolve_tool(state["tools"], "sellers")
        items_raw = await _fetch(state, items_tool, "item-agent", "item", order_id=order_id)
        sellers_raw = await _fetch(state, sellers_tool, "item-agent", "seller", order_id=order_id)
        item_rows = _as_list(items_raw, "items", "order_items")
        seller_rows = _as_list(sellers_raw, "sellers")
        has_item_evidence = bool(item_rows or seller_rows)
        trace.emit(
            case_id=case_id,
            event_type="handoff",
            actor="item-agent",
            target="payment_agent",
            decision_code=(
                "item_evidence_collected" if has_item_evidence else "item_evidence_missing"
            ),
        )
        return {
            "item_data": item_rows,
            "seller_data": seller_rows,
            "evidence_refs": state["evidence_refs"],
            "refs_by_domain": state["refs_by_domain"],
        }

    async def payment_agent(state: State) -> dict[str, Any]:
        trace.emit(
            case_id=case_id,
            event_type="task_assigned",
            actor="coordinator",
            target="payment_agent",
            decision_code="collect_payment_evidence",
        )
        order_id = state.get("order_id")
        order_data = state.get("order_data") or {}
        payment_data = await _fetch(
            state,
            _resolve_tool(state["tools"], "order_payments"),
            "payment-agent",
            "payment",
            order_id=order_id,
            payment_reference=_pick(order_data, "payment_reference", "payment_id"),
        )
        timeline_data = await _fetch(
            state,
            _resolve_tool(state["tools"], "payment_timeline"),
            "payment-agent",
            "payment_timeline",
            order_id=order_id,
        )
        payment_decision = (
            "payment_evidence_collected" if payment_data else "payment_evidence_missing"
        )
        trace.emit(
            case_id=case_id,
            event_type="handoff",
            actor="payment-agent",
            target="shipment_agent",
            decision_code=payment_decision,
        )
        return {
            "payment_data": payment_data or {},
            "payment_timeline_data": timeline_data or {},
            "evidence_refs": state["evidence_refs"],
            "refs_by_domain": state["refs_by_domain"],
        }

    async def shipment_agent(state: State) -> dict[str, Any]:
        trace.emit(
            case_id=case_id,
            event_type="task_assigned",
            actor="coordinator",
            target="shipment_agent",
            decision_code="collect_shipment_evidence",
        )
        order_data = state.get("order_data") or {}
        data = await _fetch(
            state,
            _resolve_tool(state["tools"], "shipment_summary"),
            "shipment-agent",
            "shipment",
            order_id=state.get("order_id"),
            shipment_id=_pick(order_data, "shipment_id"),
        )
        data = data or {}
        trace.emit(
            case_id=case_id,
            event_type="handoff",
            actor="shipment-agent",
            target="refund_agent",
            decision_code="shipment_evidence_collected" if data else "shipment_evidence_missing",
        )
        return {
            "shipment_data": data,
            "evidence_refs": state["evidence_refs"],
            "refs_by_domain": state["refs_by_domain"],
        }

    async def refund_agent(state: State) -> dict[str, Any]:
        trace.emit(
            case_id=case_id,
            event_type="task_assigned",
            actor="coordinator",
            target="refund_agent",
            decision_code="collect_refund_evidence",
        )
        data = await _fetch(
            state,
            _resolve_tool(state["tools"], "refund_timeline"),
            "refund-agent",
            "refund",
            order_id=state.get("order_id"),
        )
        data = data or {}
        trace.emit(
            case_id=case_id,
            event_type="handoff",
            actor="refund-agent",
            target="policy_agent",
            decision_code="refund_evidence_collected" if data else "refund_evidence_missing",
        )
        return {
            "refund_data": data,
            "evidence_refs": state["evidence_refs"],
            "refs_by_domain": state["refs_by_domain"],
        }

    async def policy_agent(state: State) -> dict[str, Any]:
        trace.emit(
            case_id=case_id,
            event_type="task_assigned",
            actor="coordinator",
            target="policy_agent",
            decision_code="collect_policy_evidence",
        )
        data = await _fetch(
            state,
            _resolve_tool(state["tools"], "policy"),
            "policy-agent",
            "policy",
            policy_version=case.get("policy_version"),
        )
        data = data or {}
        conclusion = _build_conclusion(
            case,
            order_data=state.get("order_data") or {},
            item_rows=state.get("item_data") or [],
            seller_rows=state.get("seller_data") or [],
            payment_data=state.get("payment_data") or {},
            payment_timeline_data=state.get("payment_timeline_data") or {},
            shipment_data=state.get("shipment_data") or {},
            refund_data=state.get("refund_data") or {},
            policy_data=data,
            order_id=state.get("order_id"),
        )
        trace.emit(
            case_id=case_id,
            event_type="policy_decided",
            actor="policy-agent",
            decision_code=conclusion["assessment"]["primary_issue"],
        )
        return {
            "policy_data": data,
            "final_answer": conclusion,
            "evidence_refs": state["evidence_refs"],
            "refs_by_domain": state["refs_by_domain"],
        }

    async def verifier_agent(state: State) -> dict[str, Any]:
        conclusion = dict(state["final_answer"])
        refs = state.get("evidence_refs", [])
        by_domain = state.get("refs_by_domain", {})
        conclusion["evidence_refs"] = sorted(set(refs))

        primary_issue = conclusion["assessment"]["primary_issue"]
        primary_domains = set(TOPIC_DOMAINS.get(primary_issue, ()))
        for claim in conclusion.get("claim_assessments", []):
            topic = claim.pop("_topic", None)
            if claim["verdict"] == "insufficient_evidence":
                claim["evidence_refs"] = []
                continue
            domains = set(TOPIC_DOMAINS.get(topic, ())) | primary_domains
            claim_refs = sorted({by_domain[d] for d in domains if d in by_domain})
            claim["evidence_refs"] = claim_refs or sorted(set(refs))

        refund = conclusion["financial_resolution"]["recommended_refund_brl"]
        lines_total = sum(
            line["amount_brl"] for line in conclusion["financial_resolution"]["refund_lines"]
        )
        if conclusion["financial_resolution"]["refund_lines"] and abs(lines_total - refund) > 0.01:
            conclusion["financial_resolution"]["recommended_refund_brl"] = round(lines_total, 2)
        conclusion["assessment"]["confidence"] = max(
            0.0, min(1.0, conclusion["assessment"]["confidence"])
        )
        trace.emit(
            case_id=case_id,
            event_type="verification_completed",
            actor="verifier-agent",
            decision_code=conclusion["assessment"]["case_status"],
        )
        return {"final_answer": conclusion}

    graph = StateGraph(State)
    graph.add_node("coordinator", coordinator)
    graph.add_node("order_agent", order_agent)
    graph.add_node("item_agent", item_agent)
    graph.add_node("payment_agent", payment_agent)
    graph.add_node("shipment_agent", shipment_agent)
    graph.add_node("refund_agent", refund_agent)
    graph.add_node("policy_agent", policy_agent)
    graph.add_node("verifier_agent", verifier_agent)

    graph.add_edge(START, "coordinator")
    graph.add_edge("coordinator", "order_agent")
    graph.add_edge("order_agent", "item_agent")
    graph.add_edge("item_agent", "payment_agent")
    graph.add_edge("payment_agent", "shipment_agent")
    graph.add_edge("shipment_agent", "refund_agent")
    graph.add_edge("refund_agent", "policy_agent")
    graph.add_edge("policy_agent", "verifier_agent")
    graph.add_edge("verifier_agent", END)

    app = graph.compile()
    final_state = await app.ainvoke({"case": case})
    return final_state["final_answer"]


def _build_conclusion(
    case: dict[str, Any],
    order_data: dict[str, Any],
    item_rows: list[dict[str, Any]],
    seller_rows: list[dict[str, Any]],
    payment_data: dict[str, Any],
    payment_timeline_data: dict[str, Any],
    shipment_data: dict[str, Any],
    refund_data: dict[str, Any],
    policy_data: dict[str, Any] | None,
    order_id: str | None,
) -> dict[str, Any]:
    order_status = str(_pick(order_data, "order_status", "status") or "").lower()
    payment_total = _payment_total(payment_data)
    items = _items_summary(item_rows)
    order_total = items["items_total"]
    if order_total is None:
        order_total = _num(_pick(order_data, "total_value", "order_total", "amount", "price_total"))
    has_payment = bool(payment_total and payment_total > 0)

    delivered_at = _date(
        _pick(shipment_data, "delivered_at", "order_delivered_customer_date")
        or _pick(order_data, "order_delivered_customer_date")
    )
    estimated_at = _date(
        _pick(order_data, "order_estimated_delivery_date", "estimated_delivery_date")
        or _pick(shipment_data, "estimated_delivery_date")
    )
    carrier_at = _date(
        _pick(shipment_data, "shipped_at", "order_delivered_carrier_date")
        or _pick(order_data, "order_delivered_carrier_date")
    )
    shipping_limit_keys = (
        "shipping_limit_date",
        "seller_shipping_limit_date",
        "handoff_limit_date",
    )
    shipping_limit_at = items["shipping_limit_at"] or _date(
        _pick(shipment_data, *shipping_limit_keys)
    )
    grace_days = _num(_pick(policy_data, "late_delivery_grace_days")) or 0
    is_late = bool(
        delivered_at
        and estimated_at
        and (delivered_at - estimated_at).total_seconds() > grace_days * 86400
    )
    late_owner = "logistics_provider"
    if is_late:
        never_shipped = carrier_at is None
        shipped_late = (
            bool(shipping_limit_at) and not never_shipped and carrier_at > shipping_limit_at
        )
        if never_shipped or shipped_late:
            late_owner = "seller"

    seller_row_ids = {_pick(row, "seller_id") for row in seller_rows}
    seller_ids = sorted((set(items["seller_ids"]) | seller_row_ids) - {None})
    primary_seller_id = seller_ids[0] if seller_ids else None

    payment_entries = _as_list(payment_data, "payments", "installments")
    duplicate_charge = False
    if len(payment_entries) > 1:
        amounts = [_num(_pick(entry, "payment_value", "amount")) for entry in payment_entries]
        amounts = [amount for amount in amounts if amount is not None]
        duplicate_charge = len(amounts) != len(set(amounts)) and len(amounts) > 1
    timeline_events = _as_list(payment_timeline_data, "events", "timeline", "history")
    duplicate_charge = duplicate_charge or _has_duplicate_authorization(timeline_events)

    mismatch = bool(
        payment_total is not None
        and order_total is not None
        and abs(payment_total - order_total) > 0.01
    )

    refund_events = _as_list(refund_data, "events", "timeline", "history")
    refund_status = _refund_status(refund_events)

    ranked_causes: list[dict[str, Any]] = []
    responsible_parties: list[dict[str, Any]] = []
    refund_lines: list[dict[str, Any]] = []
    resolution_actions: list[str] = []
    confidence = 0.3

    if not order_data:
        primary_issue = "insufficient_evidence"
        case_status = "needs_investigation"
        resolution_actions = ["request_additional_evidence"]
        ranked_causes = [{"cause_code": "ORDER_EVIDENCE_UNAVAILABLE", "rank": 1}]
        confidence = 0.15
    elif refund_status == "failed":
        primary_issue = "refund_failed"
        case_status = "action_required"
        responsible_parties = [{"party_type": "payment_provider", "party_id": None}]
        ranked_causes = [{"cause_code": "REFUND_ATTEMPT_FAILED", "rank": 1}]
        resolution_actions = ["retry_refund", "escalate_to_payment_provider"]
        refund_lines = (
            [
                {
                    "reason_code": "refund_failed",
                    "amount_brl": round(payment_total, 2),
                    "entity_id": order_id,
                }
            ]
            if has_payment
            else []
        )
        confidence = 0.75 if refund_events else 0.5
    elif refund_status == "pending":
        primary_issue = "refund_pending"
        case_status = "action_required"
        responsible_parties = [{"party_type": "platform", "party_id": None}]
        ranked_causes = [{"cause_code": "REFUND_STUCK_PENDING", "rank": 1}]
        resolution_actions = ["escalate_refund_processing", "notify_customer"]
        refund_lines = (
            [
                {
                    "reason_code": "refund_pending",
                    "amount_brl": round(payment_total, 2),
                    "entity_id": order_id,
                }
            ]
            if has_payment
            else []
        )
        confidence = 0.7 if refund_events else 0.45
    elif order_status in {"canceled", "cancelled"} and has_payment:
        primary_issue = "canceled_order_paid"
        case_status = "action_required"
        responsible_parties = [{"party_type": "platform", "party_id": None}]
        ranked_causes = [{"cause_code": "ORDER_CANCELED_AFTER_PAYMENT", "rank": 1}]
        resolution_actions = ["issue_full_refund", "notify_customer"]
        refund_lines = [
            {
                "reason_code": "canceled_order_paid",
                "amount_brl": round(payment_total, 2),
                "entity_id": order_id,
            }
        ]
        confidence = 0.75 if payment_data else 0.5
    elif order_status == "unavailable" and has_payment:
        primary_issue = "unavailable_order_paid"
        case_status = "action_required"
        responsible_parties = [{"party_type": "seller", "party_id": primary_seller_id}]
        ranked_causes = [{"cause_code": "ORDER_UNAVAILABLE_AFTER_PAYMENT", "rank": 1}]
        resolution_actions = ["issue_full_refund", "escalate_to_seller"]
        refund_lines = [
            {
                "reason_code": "unavailable_order_paid",
                "amount_brl": round(payment_total, 2),
                "entity_id": order_id,
            }
        ]
        confidence = 0.7 if payment_data else 0.45
    elif duplicate_charge:
        primary_issue = "duplicate_charge"
        case_status = "action_required"
        responsible_parties = [{"party_type": "payment_provider", "party_id": None}]
        ranked_causes = [{"cause_code": "PAYMENT_DUPLICATE_CHARGE", "rank": 1}]
        resolution_actions = ["issue_partial_refund", "escalate_to_payment_provider"]
        reference_total = order_total if order_total is not None else 0.0
        overpaid = max((payment_total or 0.0) - reference_total, 0.0)
        refund_lines = [
            {
                "reason_code": "duplicate_charge",
                "amount_brl": round(overpaid, 2),
                "entity_id": order_id,
            }
        ]
        confidence = 0.6
    elif mismatch:
        primary_issue = "payment_mismatch"
        case_status = "action_required"
        responsible_parties = [{"party_type": "payment_provider", "party_id": None}]
        ranked_causes = [{"cause_code": "PAYMENT_TOTAL_MISMATCH", "rank": 1}]
        resolution_actions = ["reconcile_payment", "notify_customer"]
        diff = abs(payment_total - order_total)
        refund_lines = (
            [
                {
                    "reason_code": "payment_mismatch",
                    "amount_brl": round(diff, 2),
                    "entity_id": order_id,
                }
            ]
            if payment_total > order_total
            else []
        )
        confidence = 0.55
    elif is_late:
        primary_issue = (
            "late_delivery_seller" if late_owner == "seller" else "late_delivery_logistics"
        )
        case_status = "action_required"
        late_party_id = primary_seller_id if late_owner == "seller" else None
        responsible_parties = [{"party_type": late_owner, "party_id": late_party_id}]
        ranked_causes = [
            {
                "cause_code": "LATE_SHIPMENT_SELLER"
                if late_owner == "seller"
                else "LATE_SHIPMENT_CARRIER",
                "rank": 1,
            }
        ]
        resolution_actions = [
            "notify_customer",
            "escalate_to_seller" if late_owner == "seller" else "escalate_to_logistics_partner",
        ]
        confidence = 0.65 if shipment_data else 0.4
    elif len(payment_entries) > 1 and not mismatch:
        primary_issue = "valid_split_payment"
        case_status = "no_action"
        ranked_causes = [{"cause_code": "SPLIT_PAYMENT_CONFIRMED_VALID", "rank": 1}]
        resolution_actions = ["close_case_no_action"]
        confidence = 0.6
    else:
        primary_issue = "unsupported_claim"
        case_status = "no_action"
        ranked_causes = [{"cause_code": "NO_EVIDENCE_SUPPORTS_CLAIM", "rank": 1}]
        resolution_actions = ["close_case_no_action"]
        confidence = 0.5

    claims = case.get("customer_request", {}).get("claims", [])
    claim_assessments = []
    for claim in claims[:5]:
        topic = claim.get("topic")
        if not order_data:
            verdict = "insufficient_evidence"
            claim_confidence = 0.15
        elif topic == primary_issue:
            verdict = "supported"
            claim_confidence = confidence
        elif topic == "requested_full_refund":
            verdict = "supported" if refund_lines else "unsupported"
            claim_confidence = confidence if refund_lines else max(0.2, 1 - confidence)
        else:
            verdict = "unsupported"
            claim_confidence = max(0.2, 1 - confidence)
        claim_assessments.append(
            {
                "claim_id": claim["claim_id"],
                "verdict": verdict,
                "confidence": round(max(0.0, min(1.0, claim_confidence)), 2),
                "evidence_refs": [],
                "_topic": topic,
            }
        )

    data_conflicts: list[dict[str, Any]] = []
    order_delivered_raw = _pick(order_data, "order_delivered_customer_date")
    shipment_delivered_raw = _pick(shipment_data, "delivered_at", "order_delivered_customer_date")
    order_delivered_dt = _date(order_delivered_raw)
    shipment_delivered_dt = _date(shipment_delivered_raw)
    if (
        order_delivered_dt
        and shipment_delivered_dt
        and abs((order_delivered_dt - shipment_delivered_dt).total_seconds()) > 86400
    ):
        data_conflicts.append(
            {
                "field": "order_delivered_customer_date",
                "sources": ["order", "shipment"],
                "selected_source": "shipment",
                "resolution_code": "prefer_shipment_authoritative_timestamp",
            }
        )

    entities = {
        "order_ids": sorted({order_id} - {None}) if order_id else [],
        "item_ids": items["item_ids"],
        "seller_ids": seller_ids,
        "payment_references": sorted(
            {
                str(_pick(entry, "payment_id", "payment_reference"))
                for entry in payment_entries
                if _pick(entry, "payment_id", "payment_reference") is not None
            }
        ),
        "shipment_ids": sorted({sid for sid in [_pick(order_data, "shipment_id")] if sid}),
    }

    recommended_refund = round(sum(line["amount_brl"] for line in refund_lines), 2)

    return {
        "schema_version": "day09-l3a-output-v2",
        "case_id": case["case_id"],
        "assessment": {
            "primary_issue": primary_issue,
            "case_status": case_status,
            "confidence": round(confidence, 2),
        },
        "affected_entities": entities,
        "claim_assessments": claim_assessments,
        "root_cause_analysis": {
            "ranked_causes": ranked_causes,
            "responsible_parties": responsible_parties,
        },
        "evidence_refs": [],
        "data_conflicts": data_conflicts,
        "financial_resolution": {
            "currency": "BRL",
            "recommended_refund_brl": recommended_refund,
            "refund_lines": refund_lines,
        },
        "resolution_actions": resolution_actions,
    }
