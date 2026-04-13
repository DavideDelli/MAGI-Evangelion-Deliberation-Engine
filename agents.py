import os
import time
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from utils import is_rate_limit

def _make_llm(model: str, temperature: float) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=os.getenv("GITHUB_TOKEN"),
        base_url="https://models.inference.ai.azure.com",
        model=model,
        temperature=temperature,
    )

async def ask_agent_with_fallback(
    persona_desc: str,
    dilemma: str,
    configs: list[dict],
    name: str,
) -> tuple[str, float, str]:
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            f"{persona_desc}\n"
            "You must analyze the proposed dilemma.\n"
            "At the end of your analysis, you MUST explicitly write on a new line: "
            "'VOTO: SI' or 'VOTO: NO'."
        ),
        ("user", "{dilemma}")
    ])

    for plan_idx, cfg in enumerate(configs):
        model_name = cfg["model"]
        temperature = cfg["temperature"]
        plan_label = "A" if plan_idx == 0 else chr(ord("B") + plan_idx - 1)

        llm = _make_llm(model_name, temperature)
        chain = prompt_template | llm

        max_transient_retries = 2
        for attempt in range(max_transient_retries + 1):
            try:
                t_start = time.perf_counter()
                response = await chain.ainvoke({"dilemma": dilemma})
                elapsed = round(time.perf_counter() - t_start, 2)
                if plan_idx > 0:
                    print(f"   ✓ {name} [Plan {plan_label} — {model_name}]: {elapsed}s")
                else:
                    print(f"   ✓ {name} [{model_name}]: {elapsed}s")
                return response.content, elapsed, model_name

            except Exception as e:
                if is_rate_limit(e):
                    if plan_idx + 1 < len(configs):
                        next_model = configs[plan_idx + 1]["model"]
                        print(
                            f"   ⚠️  {name} — {model_name}: daily quota exhausted (429). "
                            f"Switching to Plan {chr(ord('B') + plan_idx)}: {next_model}"
                        )
                    else:
                        print(f"   ❌ {name} — all fallback plans exhausted.")
                        raise RuntimeError(f"{name}: all fallback models exhausted.") from e
                    break
                elif attempt < max_transient_retries:
                    wait = 10 * (attempt + 1)
                    print(f"   ⚠️  {name} [{model_name}] error, retry {attempt + 1}/{max_transient_retries} in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    raise