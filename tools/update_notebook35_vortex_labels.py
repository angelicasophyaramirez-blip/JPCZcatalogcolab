"""Apply current event-screen figure labels to standalone Notebook 35, preserving its workflow.

Run after rebuilding Notebook 35 with its original builder. The label helper is shared
with Notebook 36's source so both notebooks agree on completed/unchecked scan states.
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "notebooks/35_imerg_manual_case_review.ipynb"
notebook = json.loads(path.read_text())
module_source = (ROOT / "src/jpcz_catalog/weathernext_atlas.py").read_text()
tree = ast.parse(module_source)
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "vortex_classification")
helper = ast.get_source_segment(module_source, node)
scheme = "notebook35_vortex_screen_v2"

for cell in notebook["cells"]:
    source = "".join(cell.get("source", []))
    if cell["cell_type"] == "code" and "def get_case(" in source:
        if "def vortex_classification(" not in source:
            source = source.replace("def atomic_csv(", f"VORTEX_LABEL_SCHEME = {scheme!r}\n\n{helper}\n\n\ndef atomic_csv(", 1)
        else:
            old_node = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='vortex_classification')
            source = source.replace(ast.get_source_segment(source,old_node),helper,1)
        source = source.replace(
            "flag = (' | vortex detected' if row.get('vortex_scan_status') == 'complete'\n"
            "            and row.get('vortex_detected', False) == True else '')",
            "flag = ' | ' + vortex_classification(row)")
        source = source.replace("label=label or row.get('case_label', '') or 'Unclassified candidate', **imerg_info)",
                                "legacy_manual_label=label or row.get('case_label', ''), **imerg_info)")
        anchor = "        print('Vortex screen unavailable; maps remain usable:',metadata['vortex_screen_error'])\n"
        if "metadata.update(label=vortex_classification(metadata)" not in source:
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
        assert "Unclassified candidate" not in source
        assert "def vortex_classification(" in source and "'active_screen.json'" in source
    elif cell["cell_type"] == "code" and "def compare_pins" in source:
        source = source.replace("_running_v8_vortex_screen_", "_running_v9_vortex_labels_v2_")
        source = source.replace("description='My label:'", "description='Legacy label:'")
        source = source.replace("{value['label'] or 'unclassified candidate'}", "{vortex_classification(value['row'])}")
        source = source.replace("f\"Case {key}: {pins[key]['label'] or 'unclassified candidate'}\"", "f\"Case {key}: {vortex_classification(pins[key]['row'])}\"")
    elif cell["cell_type"] == "markdown" and source.startswith("# JPCZ manual case review"):
        if "**Figure classification labels:**" not in source:
            source += ("\n\n**Figure classification labels:** figure titles and the event dropdown use the same event-level vortex screen: "
                       "**Vortex** when a completed scan detects a vortex, **Non-vortex** when a completed scan detects none, "
                       "and **Vortex not checked** for unscanned events. Failed/unknown scans never become non-vortex. "
                       "Legacy manual labels are retained as metadata but do not determine figure titles. "
                       "Click **Refresh catalog**, select the event, and **Load maps** to export newly labeled PNG/PDF figures. "
                       "Existing saved images retain their old embedded text. New exports use a new filename version. "
                       "Notebook 36 reads the same saved event-screen records; this class is observed-event context, not a forecast-member classification.\n")
    if cell["cell_type"] == "code":
        compile(source, 'notebook35_cell', 'exec')
    cell["source"] = source.splitlines(keepends=True)

path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(path)
