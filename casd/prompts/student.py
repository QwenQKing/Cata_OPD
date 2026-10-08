SOLVE_ONLY = (
    "You are a helpful assistant. For the given question, think through the problem "
    "step by step inside <think> ... </think>, then put ONLY the final answer inside "
    "<answer> ... </answer>."
)

STUDENT_INSTRUCTION = (
    SOLVE_ONLY + "\n"
    "- If your answer is incorrect, a teacher will respond inside <guidance> ... </guidance> "
    "with a step-by-step method for solving this type of problem. Learn the teacher's solving "
    "method and reasoning approach, then produce the final correct answer to this problem "
    "(output: <think> ... </think> <answer> ... </answer>)."
)

def build_solve_prompt(question: str, deploy: bool = False) -> str:

    _ = deploy
    return STUDENT_INSTRUCTION + "\n\nQuestion: " + question

def wrap_guidance(guidance: str) -> str:

    return "<guidance>" + guidance + "</guidance>"
