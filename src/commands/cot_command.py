def execute(query, use_cot):
    """Executes the /cot command to toggle Chain-of-Thought reasoning."""
    parts = query.split()
    if len(parts) == 1:
        # Toggle between disabled and default CoT
        use_cot = not use_cot
        cot_mode = "default"
        status = "ENABLED" if use_cot else "DISABLED"
        print(f"🔄 Chain-of-Thought (CoT) Reasoning {status} (Mode: {cot_mode})")
    else:
        cot_mode = parts[1]
        use_cot = True
        print(f"🔄 Chain-of-Thought (CoT) Reasoning ENABLED (Mode: {cot_mode})")
    return use_cot
