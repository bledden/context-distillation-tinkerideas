"""
Context Generation for Distillation.

Generates different types of context for the teacher model.
"""

import random
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

from config import ContextType, ContextConfig

logger = logging.getLogger(__name__)


@dataclass
class Example:
    """A single example (prompt, response) pair."""
    prompt: str
    response: str
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "metadata": self.metadata or {},
        }


@dataclass
class ContextBundle:
    """Context to provide to teacher."""
    context_type: ContextType
    context_text: str
    examples: List[Example] = None
    metadata: Dict[str, Any] = None

    def format_for_prompt(self, query: str) -> str:
        """Format context + query for teacher prompt."""
        if self.context_type == ContextType.FEW_SHOT:
            parts = ["Here are some examples:\n"]
            for ex in self.examples or []:
                parts.append(f"Q: {ex.prompt}\nA: {ex.response}\n")
            parts.append(f"\nNow answer this:\nQ: {query}\nA:")
            return "\n".join(parts)

        elif self.context_type == ContextType.RETRIEVAL:
            return f"""Relevant context:
{self.context_text}

Based on the above context, answer:
{query}"""

        elif self.context_type == ContextType.SYSTEM_INSTRUCTIONS:
            return f"""{self.context_text}

User: {query}
Assistant:"""

        elif self.context_type == ContextType.CHAIN_OF_THOUGHT:
            return f"""Solve this step by step, showing your reasoning:

{query}

Let me think through this carefully:"""

        else:
            return f"{self.context_text}\n\n{query}"


class ContextGenerator:
    """
    Generate context for teacher model.

    Supports multiple context types:
    - Few-shot examples
    - Retrieved documents
    - System instructions
    - Chain-of-thought templates
    """

    def __init__(self, config: ContextConfig):
        self.config = config
        self.example_pool: List[Example] = []
        self.retriever = None

    def load_examples(self, path: str):
        """Load example pool from file."""
        with open(path) as f:
            data = json.load(f)

        self.example_pool = [
            Example(
                prompt=d["prompt"],
                response=d["response"],
                metadata=d.get("metadata"),
            )
            for d in data
        ]
        logger.info(f"Loaded {len(self.example_pool)} examples")

    def set_retriever(self, retriever):
        """Set document retriever for retrieval context."""
        self.retriever = retriever

    def generate(
        self,
        query: str,
        context_type: Optional[ContextType] = None,
    ) -> ContextBundle:
        """Generate context for a query."""
        ctx_type = context_type or self.config.context_type

        if ctx_type == ContextType.FEW_SHOT:
            return self._generate_few_shot(query)
        elif ctx_type == ContextType.RETRIEVAL:
            return self._generate_retrieval(query)
        elif ctx_type == ContextType.SYSTEM_INSTRUCTIONS:
            return self._generate_system(query)
        elif ctx_type == ContextType.CHAIN_OF_THOUGHT:
            return self._generate_cot(query)
        elif ctx_type == ContextType.HYBRID:
            return self._generate_hybrid(query)
        else:
            raise ValueError(f"Unknown context type: {ctx_type}")

    def _generate_few_shot(self, query: str) -> ContextBundle:
        """Generate few-shot context.

        Requires an example_pool to be provided during initialization.
        """
        if not self.example_pool:
            raise ValueError(
                "Few-shot context generation requires an example_pool. "
                "Provide a list of Example objects during ContextGenerator initialization."
            )
        else:
            # Sample from pool (could use similarity-based selection)
            examples = random.sample(
                self.example_pool,
                min(self.config.num_few_shot, len(self.example_pool)),
            )

        # Format as context text
        parts = []
        for ex in examples:
            parts.append(f"Input: {ex.prompt}\nOutput: {ex.response}")

        return ContextBundle(
            context_type=ContextType.FEW_SHOT,
            context_text="\n\n".join(parts),
            examples=examples,
        )

    def _generate_retrieval(self, query: str) -> ContextBundle:
        """Generate retrieval-based context.

        Requires a retriever to be provided during initialization.
        """
        if self.retriever is None:
            raise ValueError(
                "Retrieval context generation requires a SimilarityRetriever. "
                "Provide a retriever during ContextGenerator initialization, "
                "or use a different context type (e.g., FEW_SHOT, SYSTEM_INSTRUCTIONS)."
            )

        docs = self.retriever.retrieve(query, k=self.config.retrieval_top_k)

        return ContextBundle(
            context_type=ContextType.RETRIEVAL,
            context_text="\n\n".join(docs),
            metadata={"num_docs": len(docs)},
        )

    def _generate_system(self, query: str) -> ContextBundle:
        """Generate system instruction context."""
        system_prompt = self.config.system_prompt or self._default_system_prompt()

        return ContextBundle(
            context_type=ContextType.SYSTEM_INSTRUCTIONS,
            context_text=system_prompt,
        )

    def _generate_cot(self, query: str) -> ContextBundle:
        """Generate chain-of-thought template context."""
        cot_template = """When solving problems, follow these steps:
1. Understand the question completely
2. Identify the key information
3. Break down the problem into smaller parts
4. Solve each part systematically
5. Verify your answer makes sense
6. State the final answer clearly"""

        return ContextBundle(
            context_type=ContextType.CHAIN_OF_THOUGHT,
            context_text=cot_template,
        )

    def _generate_hybrid(self, query: str) -> ContextBundle:
        """Generate hybrid context (few-shot + retrieval + instructions)."""
        few_shot = self._generate_few_shot(query)
        retrieval = self._generate_retrieval(query)
        system = self._generate_system(query)

        combined_text = f"""{system.context_text}

## Relevant Context:
{retrieval.context_text}

## Examples:
{few_shot.context_text}"""

        return ContextBundle(
            context_type=ContextType.HYBRID,
            context_text=combined_text,
            examples=few_shot.examples,
            metadata={
                "has_few_shot": True,
                "has_retrieval": True,
                "has_system": True,
            },
        )

    def _default_system_prompt(self) -> str:
        """Default system prompt for instruction-following."""
        return """You are a helpful, harmless, and honest assistant.

Guidelines:
- Provide accurate and relevant information
- Be concise but complete
- If you're unsure, say so
- Follow the user's instructions carefully
- Be respectful and professional"""


class SimilarityRetriever:
    """
    Simple similarity-based retriever using embeddings.
    """

    def __init__(self, documents: List[str] = None):
        self.documents = documents or []
        self.embeddings = None
        self._encoder = None

    def _get_encoder(self):
        """Lazy load encoder."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                logger.warning("sentence-transformers not available")
                return None
        return self._encoder

    def index(self, documents: List[str]):
        """Index documents for retrieval."""
        self.documents = documents
        encoder = self._get_encoder()
        if encoder:
            self.embeddings = encoder.encode(documents)
        logger.info(f"Indexed {len(documents)} documents")

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Retrieve top-k similar documents."""
        if not self.documents:
            return []

        encoder = self._get_encoder()
        if encoder is None or self.embeddings is None:
            # Fallback to random
            return random.sample(self.documents, min(k, len(self.documents)))

        # Compute similarity
        query_embedding = encoder.encode([query])[0]

        import numpy as np
        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-k:][::-1]

        return [self.documents[i] for i in top_indices]


class DynamicExampleSelector:
    """
    Select few-shot examples dynamically based on similarity to query.
    """

    def __init__(self, examples: List[Example]):
        self.examples = examples
        self.embeddings = None
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                return None
        return self._encoder

    def index(self):
        """Compute embeddings for all examples."""
        encoder = self._get_encoder()
        if encoder:
            prompts = [ex.prompt for ex in self.examples]
            self.embeddings = encoder.encode(prompts)

    def select(self, query: str, k: int = 5) -> List[Example]:
        """Select k most similar examples."""
        if not self.examples:
            return []

        encoder = self._get_encoder()
        if encoder is None or self.embeddings is None:
            return random.sample(self.examples, min(k, len(self.examples)))

        import numpy as np
        query_embedding = encoder.encode([query])[0]
        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-k:][::-1]

        return [self.examples[i] for i in top_indices]
