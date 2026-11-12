"""Apply current event-screen figure labels to standalone Notebook 35, preserving its workflow.

Run after rebuilding Notebook 35 with its original builder. The label functions are
shared with Notebook 36 so both notebooks use the same association labels.
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "notebooks/35_imerg_manual_case_review.ipynb"
notebook = json.loads(path.read_text())
module_source = (ROOT / "src/jpcz_catalog/weathernext_atlas.py").read_text()
tree = ast.parse(module_source)
helper_names = {"vortex_classification", "vortex_plot_label"}
helpers = {n.name: ast.get_source_segment(module_source, n)
           for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in helper_names}
scheme_node = next(n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "VORTEX_LABEL_SCHEME" for t in n.targets))
scheme = ast.literal_eval(scheme_node.value)

for cell in notebook["cells"]:
    source = "".join(cell.get("source", []))
    if cell["cell_type"] == "code" and "def get_case(" in source:
        if "VORTEX_LABEL_SCHEME = " not in source:
            source = source.replace("def atomic_csv(", f"VORTEX_LABEL_SCHEME = {scheme!r}\n\n\ndef atomic_csv(", 1)
        source = source.replace("VORTEX_LABEL_SCHEME = 'notebook35_vortex_screen_v2'", f"VORTEX_LABEL_SCHEME = {scheme!r}")
        for name, helper in helpers.items():
            old_node = next((n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name), None)
            if old_node is None:
                source = source.replace("def atomic_csv(", helper + "\n\n\ndef atomic_csv(", 1)
            else:
                source = source.replace(ast.get_source_segment(source, old_node), helper, 1)
        source = source.replace(
            "flag = (' | vortex detected' if row.get('vortex_scan_status') == 'complete'\n"
            "            and row.get('vortex_detected', False) == True else '')",
            "flag = ' | ' + vortex_classification(row)")
        source = source.replace("label=label or row.get('case_label', '') or 'Unclassified candidate', **imerg_info)",
                                "legacy_manual_label=label or row.get('case_label', ''), **imerg_info)")
        anchor = "        print('Vortex screen unavailable; maps remain usable:',metadata['vortex_screen_error'])\n"
        if not any(f"metadata.update(label={name}(metadata)" in source for name in helper_names):
            source = source.replace(anchor, anchor +
                "    metadata.update(label=vortex_classification(metadata), label_scheme=VORTEX_LABEL_SCHEME)\n"
                "    comparison = re.match(r'^Case ([AB]):', str(label or ''))\n"
                "    metadata['comparison_slot'] = comparison.group(1) if comparison else ''\n")
        source = source.replace("        header = f\"{meta['label']} | {meta['view_hour_utc']:%Y-%m-%d %H:%M} UTC\"",
            "        meta.update(label=vortex_classification(meta), label_scheme=VORTEX_LABEL_SCHEME)\n"
            "        prefix = f\"Case {meta['comparison_slot']} | \" if meta.get('comparison_slot') else ''\n"
            "        number = f\"Event {int(meta['event_number'])} | \" if pd.notna(meta.get('event_number')) else ''\n"
            "        header = prefix + f\"{meta['label']} | \" + number + f\"{meta['view_hour_utc']:%Y-%m-%d %H:%M} UTC\"")
        source = source.replace("_running_v8_vortex_screen_", "_running_v9_vortex_labels_v2_")
        checks = ("and prior.get('label_scheme',pd.Series(dtype=str)).eq(VORTEX_LABEL_SCHEME).all()\n"
                  "                    and prior.get('label',pd.Series(dtype=str)).eq(vortex_classification(row)).all()\n"
                  "                    ")
        while checks + checks in source:
            source = source.replace(checks + checks,checks)
        if "prior.get('label_scheme'" not in source:
            source = source.replace("and not prior.empty and (OUTPUT_DIR / (stem+'.png')).exists()):",
                                    checks + "and not prior.empty and (OUTPUT_DIR / (stem+'.png')).exists()):")
        anchor = "    records = [cached_event_vortex(row) for _,row in frame.iterrows()]\n"
        if "'active_screen.json'" not in source:
            source = source.replace(anchor,
                "    vortex_cache_dir()  # ensure the parent exists before writing the active screen pointer\n"
                "    atomic_json(dict(signature=vortex_signature(), label_scheme=VORTEX_LABEL_SCHEME),\n"
                "                OUTPUT_DIR/'vortex_screen'/'active_screen.json')\n" + anchor)
        signature_line = "        vortex_screen_signature=result.get('signature',''),\n"
        while signature_line + signature_line in source:
            source = source.replace(signature_line + signature_line,signature_line)
        if "vortex_screen_signature=result.get('signature','')," not in source:
            source = source.replace("return dict(vortex_detected=result['vortex_detected'],vortex_scan_status=result['status'],",
                "return dict(vortex_detected=result['vortex_detected'],vortex_scan_status=result['status'],\n"
                "        vortex_screen_signature=result.get('signature',''),")
        source = source.replace("flag = ' | ' + vortex_classification(row)", "flag = ' | ' + vortex_plot_label(row)")
        source = source.replace("label=vortex_classification(", "label=vortex_plot_label(")
        source = source.replace(
            "        number = f\"Event {int(meta['event_number'])} | \" if pd.notna(meta.get('event_number')) else ''\n"
            "        header = prefix + f\"{meta['label']} | \" + number + f\"{meta['view_hour_utc']:%Y-%m-%d %H:%M} UTC\"",
            "        number = f\" | Event {int(meta['event_number'])}\" if pd.notna(meta.get('event_number')) else ''\n"
            "        header = prefix + meta['label'] + number + f\"\\n{meta['view_hour_utc']:%Y-%m-%d %H:%M} UTC\"")
        source = source.replace(".eq(vortex_classification(row))", ".eq(vortex_plot_label(row))")
        source = source.replace("_running_v9_vortex_labels_v2_", "_running_v10_vortex_labels_v3_")
        source = source.replace(
            "        metadata.update(vortex_scan_status='unavailable',vortex_detected=None,\n"
            "            vortex_map_centers='[]',vortex_map_status='unavailable',",
            "        metadata.update(vortex_scan_status=row.get('vortex_scan_status', 'unavailable'),\n"
            "            vortex_detected=row.get('vortex_detected'),\n"
            "            vortex_map_centers='[]',vortex_map_status='unavailable',")
        source = source.replace("ax.set_title(f'{platform} {product}\\n", "ax.set_title(f\"{meta['label']}\\n\" + f'{platform} {product}\\n")
        assert "Unclassified candidate" not in source
        assert "def vortex_classification(" in source and "'active_screen.json'" in source
    elif cell["cell_type"] == "code" and "def compare_pins" in source:
        source = source.replace("_running_v8_vortex_screen_", "_running_v9_vortex_labels_v2_")
        source = source.replace("description='My label:'", "description='Legacy label:'")
        source = source.replace("{value['label'] or 'unclassified candidate'}", "{vortex_classification(value['row'])}")
        source = source.replace("f\"Case {key}: {pins[key]['label'] or 'unclassified candidate'}\"", "f\"Case {key}: {vortex_classification(pins[key]['row'])}\"")
        source = source.replace("_running_v9_vortex_labels_v2_", "_running_v10_vortex_labels_v3_")
        source = source.replace("vortex_classification(value['row'])", "vortex_plot_label(value['row'])")
        source = source.replace("vortex_classification(pins[key]['row'])", "vortex_plot_label(pins[key]['row'])")
    elif cell["cell_type"] == "markdown" and source.startswith("# JPCZ manual case review"):
        source = source.split("\n\n**Figure classification labels", 1)[0].rstrip()
        source += ("\n\n**Figure classification labels** follow the event selector. A saved positive vortex result displays "
                   "**Vortex-associated**. Events without that flag display **Non-vortex-associated**. "
                   "The event details and export metadata record the scan status. "
                   "To regenerate a figure, click **Refresh catalog**, select the event and time, and click **Load maps**. "
                   "For the −12 h, peak, and +12 h plots, click **Show 3 times**.\n")
    if cell["cell_type"] == "code":
        compile(source, 'notebook35_cell', 'exec')
    cell["source"] = source.splitlines(keepends=True)

path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(path)
