"""Context Assembler Engine: assemble LLM prompts from many sources under a hard, verified token budget."""

from .assembler import AssembledContext, BudgetExceededError, ContextAssembler
from .manifest import ItemDecision, Manifest, SlotReport
from .models import ModelRegistry, ModelSpec, default_registry
from .policy import PolicyOverride, SlotPolicy, default_policies
from .schema import AssemblyRequest, Budget, ContextItem, FewShotExample, ToolSpec, Turn
from .slots import SlotKind
from .summarise import ExtractiveSummariser, LLMSummariser, Summariser, select_summariser
from .tokenizer import ApproxTokenizer, ChatFormat, TiktokenTokenizer, Tokenizer, count_prompt, get_tokenizer

__version__ = "0.1.0"

__all__ = [
    "ApproxTokenizer",
    "AssembledContext",
    "AssemblyRequest",
    "Budget",
    "BudgetExceededError",
    "ChatFormat",
    "ContextAssembler",
    "ContextItem",
    "ExtractiveSummariser",
    "FewShotExample",
    "ItemDecision",
    "LLMSummariser",
    "Manifest",
    "ModelRegistry",
    "ModelSpec",
    "PolicyOverride",
    "SlotKind",
    "SlotPolicy",
    "SlotReport",
    "Summariser",
    "TiktokenTokenizer",
    "Tokenizer",
    "ToolSpec",
    "Turn",
    "__version__",
    "count_prompt",
    "default_policies",
    "default_registry",
    "get_tokenizer",
    "select_summariser",
]
