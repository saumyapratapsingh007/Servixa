from __future__ import annotations

from typing import Dict, List


TASKS: List[Dict[str, object]] = [
    {
        "id": "easy_password_and_shipping",
        "difficulty": "easy",
        "title": "Retail Inbox Starter Queue",
        "objective": (
            "Triage two standard support tickets. Correctly classify each issue, "
            "set the right priority, send the best response template, and close only "
            "the ticket that can be fully resolved by support."
        ),
        "max_steps": 7,
        "guidance": [
            "Security issues must never be closed without routing to the security queue.",
            "Shipping complaints waiting on a carrier should stay open and be routed to logistics.",
            "Choose one of the visible response templates for every handled ticket.",
        ],
        "tickets": [
            {
                "ticket_id": "E-101",
                "customer_name": "Alicia Perez",
                "subject": "Locked out after too many login attempts",
                "body": (
                    "I was trying to log into my storefront account and now it says my "
                    "password needs to be reset. Please help because payroll closes in two hours."
                ),
                "channel": "email",
                "customer_tier": "standard",
                "order_value": 0.0,
                "hours_open": 3,
                "sla_hours_remaining": 5,
                "sentiment": "frustrated",
                "prior_contacts": 1,
                "tags": ["login", "account"],
                "visible_notes": ["Customer passed identity verification yesterday."],
                "allowed_templates": [
                    "password_reset_instructions",
                    "shipping_delay_empathy",
                    "billing_refund_acknowledgement",
                ],
                "expected_category": "account_access",
                "expected_priority": "high",
                "expected_route": "frontline",
                "expected_template": "password_reset_instructions",
                "expected_resolution": "reset_link_sent",
                "must_close": True,
            },
            {
                "ticket_id": "E-102",
                "customer_name": "Marcus Lee",
                "subject": "Order 55718 still not delivered",
                "body": (
                    "Tracking has not moved in four days and this was a birthday gift. "
                    "Can someone check with the carrier before tomorrow?"
                ),
                "channel": "chat",
                "customer_tier": "standard",
                "order_value": 84.99,
                "hours_open": 14,
                "sla_hours_remaining": 10,
                "sentiment": "concerned",
                "prior_contacts": 0,
                "tags": ["shipping", "carrier_delay"],
                "visible_notes": ["Warehouse already confirmed the parcel left on time."],
                "allowed_templates": [
                    "shipping_delay_empathy",
                    "password_reset_instructions",
                    "vip_outage_update",
                ],
                "expected_category": "shipping",
                "expected_priority": "medium",
                "expected_route": "logistics",
                "expected_template": "shipping_delay_empathy",
                "expected_resolution": "awaiting_carrier_followup",
                "must_close": False,
            },
        ],
    },
    {
        "id": "medium_refund_policy_mix",
        "difficulty": "medium",
        "title": "Refunds and Policy Exceptions",
        "objective": (
            "Handle a mixed queue involving a refund, a duplicate charge investigation, "
            "and an abuse report. Prioritize correctly, route high-risk work to the right "
            "teams, and avoid closing anything that still needs specialist review."
        ),
        "max_steps": 10,
        "guidance": [
            "Duplicate charges belong with billing, not the frontline queue.",
            "Trust and safety complaints require elevated priority when abuse is ongoing.",
            "Partial progress matters, but closing the wrong ticket is penalized.",
        ],
        "tickets": [
            {
                "ticket_id": "M-201",
                "customer_name": "Riya Kapoor",
                "subject": "Need refund for duplicate order",
                "body": (
                    "I accidentally checked out twice and both orders shipped. I only want one "
                    "and I have already refused the second package."
                ),
                "channel": "email",
                "customer_tier": "standard",
                "order_value": 129.50,
                "hours_open": 6,
                "sla_hours_remaining": 18,
                "sentiment": "neutral",
                "prior_contacts": 1,
                "tags": ["refund", "duplicate_order"],
                "visible_notes": ["Carrier confirms package refusal scan."],
                "allowed_templates": [
                    "billing_refund_acknowledgement",
                    "trust_safety_report_received",
                    "duplicate_charge_escalation",
                ],
                "expected_category": "billing",
                "expected_priority": "medium",
                "expected_route": "billing",
                "expected_template": "billing_refund_acknowledgement",
                "expected_resolution": "refund_issued",
                "must_close": True,
            },
            {
                "ticket_id": "M-202",
                "customer_name": "Noah Bennett",
                "subject": "My card was charged twice for one subscription",
                "body": (
                    "I only have one active workspace but my credit card shows two charges today. "
                    "Please investigate before the pending charge settles."
                ),
                "channel": "email",
                "customer_tier": "pro",
                "order_value": 299.00,
                "hours_open": 2,
                "sla_hours_remaining": 6,
                "sentiment": "anxious",
                "prior_contacts": 0,
                "tags": ["billing", "duplicate_charge"],
                "visible_notes": ["Potential payment processor issue flagged by monitoring."],
                "allowed_templates": [
                    "duplicate_charge_escalation",
                    "billing_refund_acknowledgement",
                    "password_reset_instructions",
                ],
                "expected_category": "billing",
                "expected_priority": "high",
                "expected_route": "billing",
                "expected_template": "duplicate_charge_escalation",
                "expected_resolution": "billing_investigation_opened",
                "must_close": False,
            },
            {
                "ticket_id": "M-203",
                "customer_name": "Sofia Ramirez",
                "subject": "Seller keeps sending abusive messages",
                "body": (
                    "I reported this twice already. The seller is still messaging threats through "
                    "the platform and I want this escalated today."
                ),
                "channel": "chat",
                "customer_tier": "standard",
                "order_value": 56.25,
                "hours_open": 9,
                "sla_hours_remaining": 3,
                "sentiment": "angry",
                "prior_contacts": 2,
                "tags": ["abuse_report", "safety"],
                "visible_notes": ["Previous report ID TS-778 is attached."],
                "allowed_templates": [
                    "trust_safety_report_received",
                    "shipping_delay_empathy",
                    "vip_outage_update",
                ],
                "expected_category": "trust_safety",
                "expected_priority": "urgent",
                "expected_route": "trust_safety",
                "expected_template": "trust_safety_report_received",
                "expected_resolution": "trust_safety_escalated",
                "must_close": False,
                "unsafe_if_closed_early": True,
            },
        ],
    },
    {
        "id": "hard_security_vip_outage",
        "difficulty": "hard",
        "title": "VIP Incident and Security Queue",
        "objective": (
            "Coordinate a high-pressure queue containing a possible account compromise, "
            "a VIP outage complaint, a legal data request, and a refund dispute. The goal "
            "is to triage safely, prioritize urgent work, and preserve specialist review paths."
        ),
        "max_steps": 13,
        "guidance": [
            "Possible account compromise must be routed to security with urgent priority.",
            "Legal requests should not be closed by support and belong with trust_safety.",
            "VIP operational outages should be acknowledged quickly and routed to tech_ops.",
            "Efficiency matters: unnecessary steps reduce reward even if the final grader is high.",
        ],
        "tickets": [
            {
                "ticket_id": "H-301",
                "customer_name": "Evelyn Stone",
                "subject": "Someone changed my payout bank account",
                "body": (
                    "I got an email saying my payout details were changed, but I did not do that. "
                    "Please lock the account immediately because this looks fraudulent."
                ),
                "channel": "email",
                "customer_tier": "pro",
                "order_value": 1480.00,
                "hours_open": 1,
                "sla_hours_remaining": 2,
                "sentiment": "panicked",
                "prior_contacts": 0,
                "tags": ["security", "account_compromise", "payments"],
                "visible_notes": ["Automated risk engine also flagged a new device login."],
                "allowed_templates": [
                    "security_lockdown_notice",
                    "vip_outage_update",
                    "legal_request_acknowledgement",
                ],
                "expected_category": "security",
                "expected_priority": "urgent",
                "expected_route": "security",
                "expected_template": "security_lockdown_notice",
                "expected_resolution": "security_escalation_opened",
                "must_close": False,
                "unsafe_if_closed_early": True,
            },
            {
                "ticket_id": "H-302",
                "customer_name": "Northwind Design",
                "subject": "VIP storefront is down during campaign launch",
                "body": (
                    "Our storefront returns a 503 for checkout and we are paying for a managed enterprise plan. "
                    "Please involve engineering right now."
                ),
                "channel": "phone",
                "customer_tier": "vip",
                "order_value": 12500.00,
                "hours_open": 1,
                "sla_hours_remaining": 1,
                "sentiment": "urgent",
                "prior_contacts": 1,
                "tags": ["outage", "checkout", "vip"],
                "visible_notes": ["Status page shows elevated checkout latency."],
                "allowed_templates": [
                    "vip_outage_update",
                    "shipping_delay_empathy",
                    "billing_refund_acknowledgement",
                ],
                "expected_category": "technical_outage",
                "expected_priority": "urgent",
                "expected_route": "tech_ops",
                "expected_template": "vip_outage_update",
                "expected_resolution": "incident_escalated",
                "must_close": False,
            },
            {
                "ticket_id": "H-303",
                "customer_name": "M. Chen",
                "subject": "Formal request for user data export",
                "body": (
                    "I am requesting a complete export of all messages and transaction records linked to my account "
                    "for an active legal dispute. Please confirm receipt."
                ),
                "channel": "email",
                "customer_tier": "standard",
                "order_value": 0.0,
                "hours_open": 20,
                "sla_hours_remaining": 8,
                "sentiment": "formal",
                "prior_contacts": 0,
                "tags": ["legal", "data_request"],
                "visible_notes": ["Identity check not yet completed."],
                "allowed_templates": [
                    "legal_request_acknowledgement",
                    "password_reset_instructions",
                    "trust_safety_report_received",
                ],
                "expected_category": "legal_request",
                "expected_priority": "high",
                "expected_route": "trust_safety",
                "expected_template": "legal_request_acknowledgement",
                "expected_resolution": "legal_review_queued",
                "must_close": False,
            },
            {
                "ticket_id": "H-304",
                "customer_name": "Jamal Torres",
                "subject": "Refund still missing after cancellation",
                "body": (
                    "I canceled within the trial period, but the charge posted anyway and support promised a refund "
                    "three days ago."
                ),
                "channel": "chat",
                "customer_tier": "standard",
                "order_value": 49.00,
                "hours_open": 28,
                "sla_hours_remaining": 4,
                "sentiment": "frustrated",
                "prior_contacts": 2,
                "tags": ["refund", "escalation"],
                "visible_notes": ["Refund approved internally but not processed."],
                "allowed_templates": [
                    "billing_refund_acknowledgement",
                    "duplicate_charge_escalation",
                    "security_lockdown_notice",
                ],
                "expected_category": "billing",
                "expected_priority": "high",
                "expected_route": "billing",
                "expected_template": "billing_refund_acknowledgement",
                "expected_resolution": "refund_issued",
                "must_close": True,
            },
        ],
    },
]


def get_task(task_id: str) -> Dict[str, object]:
    for task in TASKS:
        if task["id"] == task_id:
            return task
    raise KeyError(f"Unknown task_id: {task_id}")


def list_task_summaries() -> List[Dict[str, object]]:
    return [
        {
            "id": task["id"],
            "difficulty": task["difficulty"],
            "title": task["title"],
            "objective": task["objective"],
            "max_steps": task["max_steps"],
            "ticket_count": len(task["tickets"]),
        }
        for task in TASKS
    ]
