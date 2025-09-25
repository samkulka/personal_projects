import builtins

def apply_mappings(rows, table_cfg):
    # renames
    renamed = []
    colmap = table_cfg.get("columns", {})
    transforms = table_cfg.get("transforms", {})
    enums = table_cfg.get("enums", {})
    casts = table_cfg.get("casts", {})

    for r in rows:
        out = {}
        for src, tgt in colmap.items():
            out[tgt] = r.get(src)
        # enums
        for col, mapping in enums.items():
            if col in out and out[col] in mapping:
                out[col] = mapping[out[col]]
        # transforms (simple lambda eval over row)
        for col, expr in transforms.items():
            func = eval(expr, {"__builtins__": {"int": int, "round": round, "float": float}})
            out[col] = func(r)
        # casts (basic examples)
        # you can extend with dateutil for dates
        if casts:
            if "date" in casts.values():
                pass
        renamed.append(out)
    return renamed