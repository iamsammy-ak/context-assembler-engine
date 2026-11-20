"""Seeded synthetic scenarios with labelled "required facts" for the benchmark.

Each scenario is a support conversation about a faulty smart-home device (all invented). Six facts,
each carrying a unique marker string, are planted where a real assistant would need them:

========================  ============================================================
``history_early``         the customer's account number, said in the second user turn
``history_recent``        the replacement serial number, said a few turns before the end
``memory``                a loyalty code in a high-importance memory item
``retrieved_top``         a refund-policy code in one of the top-scored retrieved chunks
``retrieved_mid``         a return-label reference in a middle-scored chunk
``tool``                  the tool the query needs (a fact is kept if its schema is sent)
========================  ============================================================

The filler around them is deliberately realistic: other identifiers (tickets, orders, firmware
versions) compete with the fact sentences, near-duplicate chunks and memories are included, and the
scores the engine sees are noisy (some distractors outrank the mid fact). The engine is never told
which items hold facts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Literal

from ..schema import AssemblyRequest, Budget, ContextItem, FewShotExample, ToolSpec, Turn

FactLocation = Literal["history_early", "history_recent", "memory", "retrieved_top", "retrieved_mid", "tool"]


@dataclass(frozen=True)
class Fact:
    """A labelled fact: ``marker`` must appear in the rendered prompt (tool facts: in the tools list)."""

    id: str
    location: FactLocation
    marker: str


@dataclass(frozen=True)
class Scenario:
    seed: int
    request: AssemblyRequest
    facts: tuple[Fact, ...]


PRODUCTS = ["thermostat", "doorbell camera", "smart plug", "air purifier", "robot vacuum", "mesh router",
            "smoke detector", "baby monitor", "smart lock", "light strip"]  # fmt: skip
ISSUES = ["keeps dropping off the Wi-Fi", "shows a red status light", "will not pair with the app",
          "reboots every few hours", "stopped sending notifications", "reports the wrong temperature",
          "makes a clicking noise", "drains its battery in a day"]  # fmt: skip
ACTIONS = ["reset it to factory settings", "updated the firmware", "moved it closer to the router",
           "reinstalled the app", "changed the batteries", "tried a different power socket"]  # fmt: skip
TIMES = ["this morning", "yesterday evening", "last week", "on Monday", "after the last update", "since the weekend"]
CAUSES = ["weak Wi-Fi signal", "failed firmware update", "faulty power adapter", "pairing timeout", "hardware fault"]
TOPICS = ["pairing", "firmware updates", "Wi-Fi coverage", "battery life", "warranty claims", "shipping times",
          "account security", "subscription plans", "data privacy", "installation", "energy reports",
          "notifications", "voice assistants", "guest access"]  # fmt: skip

NEEDED_TOOLS: list[tuple[str, str, str]] = [
    ("create_return_label", "create a prepaid return label",
     "Create a prepaid return label for a faulty device so the customer can send it back for replacement."),
    ("issue_refund", "issue a refund",
     "Issue a refund for a faulty device to the original payment method after the return is received."),
    ("schedule_replacement", "schedule a replacement unit",
     "Schedule a replacement unit for a faulty device and reserve stock in the nearest warehouse."),
]  # fmt: skip
TOOL_POOL: list[tuple[str, str]] = [
    ("get_order", "Look up an order by number, with items, delivery date and payment status."),
    ("list_orders", "List the customer's recent orders, newest first."),
    ("get_device_status", "Read the live status, firmware version and signal strength of a registered device."),
    ("restart_device", "Send a remote restart command to a registered device."),
    ("push_firmware", "Push the latest firmware to a device that is online."),
    ("get_warranty", "Check warranty coverage and expiry for a device serial number."),
    ("update_address", "Change the delivery address on the customer's account."),
    ("track_shipment", "Track a parcel by its tracking reference and return the latest scan."),
    ("open_ticket", "Open a support ticket with a summary and priority."),
    ("escalate_ticket", "Escalate a ticket to the second-line hardware team."),
    ("send_email", "Send a plain-text email to the customer."),
    ("get_subscription", "Show the customer's protection-plan subscription and renewal date."),
    ("cancel_subscription", "Cancel a protection-plan subscription at the end of the period."),
    ("apply_voucher", "Apply a promotional voucher code to the next order."),
    ("search_knowledge_base", "Search help articles and return the most relevant extracts."),
    ("get_energy_report", "Return the monthly energy report for a thermostat or plug."),
    ("reset_password", "Send a password-reset link to the account email."),
    ("list_devices", "List all devices registered to the customer's home."),
    ("rename_device", "Rename a registered device in the app."),
    ("share_access", "Give another household member access to a device."),
    ("check_stock", "Check warehouse stock for a product."),
    ("refund_status", "Look up the status of an existing refund by its reference."),
    ("print_packing_slip", "Generate a packing slip PDF for an outgoing order."),
]
PARAMETERS = [
    ("order_id", "string", "Order number, e.g. ORD-10422."),
    ("serial", "string", "Device serial number as printed on the label."),
    ("device_id", "string", "Identifier of a registered device."),
    ("reason", "string", "Short reason, stored for reporting."),
    ("priority", "string", "One of low, normal or high."),
    ("email", "string", "Customer email address."),
    ("notify", "boolean", "Email the customer a confirmation."),
    ("limit", "integer", "Maximum number of results."),
]


def _code(rng: random.Random, prefix: str, digits: int) -> str:
    return f"{prefix}-{rng.randint(10 ** (digits - 1), 10**digits - 1)}"


def _user_line(rng: random.Random, product: str) -> str:
    templates = [
        lambda: f"The {product} {rng.choice(ISSUES)} {rng.choice(TIMES)}.",
        lambda: f"I already {rng.choice(ACTIONS)}, but nothing changed.",
        lambda: f"We have {rng.randint(2, 6)} of these at home and only one is affected.",
        lambda: f"It is on firmware v{rng.randint(2, 9)}.{rng.randint(0, 9)}.{rng.randint(0, 20)}, if that helps.",
        lambda: f"I opened ticket {_code(rng, 'TCK', 4)} about something similar a while ago.",
        lambda: f"Honestly it worked fine until {rng.choice(TIMES)}.",
        lambda: f"My partner also {rng.choice(ACTIONS)} {rng.choice(TIMES)}.",
    ]
    return " ".join(rng.choice(templates)() for _ in range(rng.randint(1, 3)))


def _assistant_line(rng: random.Random, product: str) -> str:
    templates = [
        lambda: f"Thanks for the details about the {product}.",
        lambda: "Could you tell me whether the status light is blinking or solid?",
        lambda: f"That behaviour usually points to a {rng.choice(CAUSES)}.",
        lambda: "Let me check the diagnostics we have for that device.",
        lambda: f"I can see order {_code(rng, 'ORD', 5)} on your account, delivered {rng.choice(TIMES)}.",
        lambda: "Please keep the device plugged in while I look into it.",
        lambda: f"Our records show the last check-in was {rng.choice(TIMES)}.",
    ]
    return " ".join(rng.choice(templates)() for _ in range(rng.randint(1, 3)))


def _article(rng: random.Random, topic: str, sentences: int) -> str:
    templates = [
        lambda: f"This article explains {topic} for {rng.choice(PRODUCTS)} owners.",
        lambda: f"Most problems with {topic} are solved after you {rng.choice(ACTIONS)}.",
        lambda: f"If the device {rng.choice(ISSUES)}, check the {rng.choice(CAUSES)} first.",
        lambda: (
            f"Customers on firmware v{rng.randint(2, 9)}.{rng.randint(0, 9)} should update before contacting support."
        ),
        lambda: f"Support can review {topic} remotely once the device is online.",
        lambda: f"Allow up to {rng.randint(2, 10)} minutes for changes to take effect.",
        lambda: f"See also the guide on {rng.choice(TOPICS)}.",
        lambda: f"Ticket references such as {_code(rng, 'TCK', 4)} help the team find earlier reports.",
    ]
    return " ".join(rng.choice(templates)() for _ in range(sentences))


def _tool(name: str, description: str, rng: random.Random) -> ToolSpec:
    chosen = rng.sample(PARAMETERS, rng.randint(1, 4))
    properties: dict[str, Any] = {param: {"type": kind, "description": text} for param, kind, text in chosen}
    return ToolSpec(
        name=name,
        description=description,
        parameters={"type": "object", "properties": properties, "required": [chosen[0][0]]},
    )


def make_scenario(seed: int, *, budget: Budget | None = None) -> Scenario:
    """A reproducible scenario for ``seed`` (same seed, same scenario, on any platform)."""
    rng = random.Random(seed)
    product = rng.choice(PRODUCTS)
    account, serial = _code(rng, "ACC", 5), f"SN-{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}"
    loyalty, policy, label = _code(rng, "LYL", 4), _code(rng, "RP", 4), _code(rng, "LBL", 5)
    tool_name, tool_phrase, tool_description = rng.choice(NEEDED_TOOLS)
    facts = (
        Fact("account", "history_early", account),
        Fact("serial", "history_recent", serial),
        Fact("loyalty", "memory", loyalty),
        Fact("refund-policy", "retrieved_top", policy),
        Fact("return-label", "retrieved_mid", label),
        Fact("needed-tool", "tool", tool_name),
    )

    # History: alternating turns; the early and recent facts are planted in user turns.
    n_turns = 2 * rng.randint(14, 20)
    history = []
    for index in range(n_turns):
        role = "user" if index % 2 == 0 else "assistant"
        text = _user_line(rng, product) if role == "user" else _assistant_line(rng, product)
        if index == 2:
            text = f"For the record, my customer account number is {account}. {text}"
        if index == n_turns - 4:
            text = f"The serial number on the replacement box is {serial}. {text}"
        history.append(Turn(role=role, content=text))  # type: ignore[arg-type]

    # Memory: importance scores are noisy; the fact item is important but not always the top.
    memory = [
        ContextItem(
            id=f"mem-{i}",
            text=f"Customer {rng.choice(['mentioned', 'noted', 'asked about', 'prefers'])} {rng.choice(TOPICS)} "
            f"for the {rng.choice(PRODUCTS)} {rng.choice(TIMES)}; {rng.choice(CAUSES)} suspected.",
            score=round(rng.uniform(0.05, 0.95), 3),
        )
        for i in range(rng.randint(18, 24))
    ]
    memory.insert(
        rng.randrange(len(memory)),
        ContextItem(
            id="mem-loyalty",
            text=f"Loyalty membership code is {loyalty}; the customer prefers store credit for small amounts.",
            score=round(rng.uniform(0.6, 0.9), 3),
        ),
    )
    for original in rng.sample(memory, 2):
        memory.append(
            ContextItem(id=f"{original.id}-copy", text=original.text.rstrip(".") + " again.", score=original.score)
        )

    # Retrieval: distractor scores overlap the mid fact's score; near-duplicate chunks are included.
    retrieved = [
        ContextItem(id=f"kb-{i}", text=_article(rng, rng.choice(TOPICS), rng.randint(3, 7)),
                    score=round(rng.uniform(0.15, 0.85), 3))
        for i in range(rng.randint(20, 26))
    ]  # fmt: skip
    retrieved.append(
        ContextItem(
            id="kb-refunds",
            text=f"Refunds for faulty devices. Under refund policy {policy}, refunds go to the original payment "
            f"method within {rng.randint(3, 10)} business days of the return being scanned. "
            + _article(rng, "warranty claims", 3),
            score=round(rng.uniform(0.86, 0.95), 3),
        )
    )
    retrieved.append(
        ContextItem(
            id="kb-returns",
            text=_article(rng, "shipping times", 2)
            + f" Replacement units ship with a prepaid return label, reference {label}, valid for 30 days. "
            + _article(rng, "installation", 2),
            score=round(rng.uniform(0.55, 0.72), 3),
        )
    )
    for original in sorted(retrieved, key=lambda item: -item.score)[2:5]:
        cut = original.text.rsplit(". ", 1)[0] + "."
        retrieved.append(ContextItem(id=f"{original.id}-dup", text=cut, score=round(original.score - 0.01, 3)))
    rng.shuffle(retrieved)

    tools = [_tool(name, description, rng) for name, description in rng.sample(TOOL_POOL, 15)]
    tools.insert(rng.randrange(len(tools)), _tool(tool_name, tool_description, rng))

    few_shot = [
        FewShotExample(
            id=f"ex-{i}",
            user=_user_line(rng, rng.choice(PRODUCTS)) + " What should I do?",
            assistant=_assistant_line(rng, rng.choice(PRODUCTS)) + " " + _article(rng, rng.choice(TOPICS), 2),
            score=round(rng.uniform(0.3, 0.9), 3),
        )
        for i in range(3)
    ]
    instructions = [
        ContextItem(id="cite", text="Cite help articles by their id in square brackets.", required=True),
        ContextItem(id="confirm", text="Confirm identifiers back to the customer before acting.", score=0.8),
        ContextItem(id="tone", text="Be friendly and brief.", score=0.4),
        ContextItem(id="escalate", text="Escalate hardware faults that persist after a reset.", score=0.6),
    ]
    query = (
        f"Can you {tool_phrase} for the faulty {product}? Please confirm my account number and the serial on "
        "the replacement box, and remind me how the refund and the return label work."
    )
    request = AssemblyRequest(
        budget=budget or Budget(context_window=8192),
        system=(
            "You are the support assistant for a smart-home device shop. Resolve device faults, returns "
            "and refunds using the account memory, help articles and tools. Be precise with identifiers."
        ),
        instructions=instructions,
        tools=tools,
        memory=memory,
        retrieved=retrieved,
        history=history,
        few_shot=few_shot,
        query=query,
    )
    return Scenario(seed=seed, request=request, facts=facts)
