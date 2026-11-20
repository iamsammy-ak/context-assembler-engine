"""Assemble a small prompt in code and print the budget report.

Run from the repository root:  python examples/quickstart.py
"""

from context_assembler import (
    AssemblyRequest,
    Budget,
    ContextAssembler,
    ContextItem,
    ToolSpec,
    Turn,
)

request = AssemblyRequest(
    budget=Budget(context_window=1024, output_reserve=256),
    system="You are a concise travel-booking assistant.",
    instructions=[ContextItem(id="confirm", text="Confirm dates before booking anything.", required=True)],
    tools=[
        ToolSpec(
            name="search_trains",
            description="Search train connections between two stations on a date.",
            parameters={
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "Departure station."},
                    "destination": {"type": "string", "description": "Arrival station."},
                    "date": {"type": "string", "description": "Travel date, YYYY-MM-DD."},
                },
                "required": ["origin", "destination", "date"],
            },
        ),
    ],
    memory=[ContextItem(id="pref", text="Prefers window seats and morning departures.", score=0.8)],
    retrieved=[
        ContextItem(
            id=f"faq-{i}",
            text=f"Fare rule {i}: tickets are refundable up to {i + 1} days before travel.",
            score=1 - i / 10,
        )
        for i in range(30)
    ],
    history=[
        Turn(role="user", content="I need to get from Turin to Milan next Friday."),
        Turn(role="assistant", content="Morning or afternoon? I can also check refundable fares."),
    ],
    query="Morning, please. Which fares can I still cancel?",
)

result = ContextAssembler().assemble(request)
manifest = result.manifest
print(f"{manifest.verified_tokens}/{manifest.prompt_budget} tokens, {manifest.overflow_repairs} overflow repairs")
for slot in manifest.slots:
    print(f"{slot.slot.label:>12}: {slot.used_tokens:4} tokens, kept {slot.items_kept}/{slot.items_in}")
kwargs = result.request_kwargs()  # messages, tools and max_tokens for an OpenAI-compatible client
print(sorted(kwargs))
