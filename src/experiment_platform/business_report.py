"""Server-side PDF memo generated from the exact UI calculation snapshot."""
from __future__ import annotations

from html import escape
from io import BytesIO
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle


def report_payload(plan: dict, snapshot: dict | None, stage: str) -> dict:
    """JSON companion also supplies the PDF. No customer IDs are accepted."""
    return {"stage": stage, "protocol": plan, "snapshot": snapshot,
            "purpose": "Traceable calculation memorandum; not evidence of completed audit or approval."}


def build_pdf(plan: dict, snapshot: dict | None = None, stage: str = "Build test plan") -> bytes:
    font = "BusinessSansSC"
    if font not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(font,str(Path(__file__).parent / "assets" / "fonts" / "BusinessSansSC-Regular.ttf")))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BusinessBody", fontName=font, fontSize=9, leading=13, textColor=colors.HexColor("#24364a"), spaceAfter=6, wordWrap="CJK"))
    styles.add(ParagraphStyle(name="BusinessSmall", parent=styles["BusinessBody"], fontSize=7, leading=10, spaceAfter=2))
    styles.add(ParagraphStyle(name="BusinessTitle", parent=styles["BusinessBody"], fontSize=23, leading=28, spaceAfter=12))
    styles.add(ParagraphStyle(name="BusinessHeading", parent=styles["BusinessBody"], fontSize=14, leading=18, spaceBefore=14, spaceAfter=8, textColor=colors.HexColor("#007e87")))
    story = []
    def para(value, style="BusinessBody"):
        text = str(value).replace("×", " x ").replace("²", "^2").replace("−", "-")
        return Paragraph(escape(text).replace("\n", "<br/>"), styles[style])
    def heading(value):
        story.append(para(value, "BusinessHeading"))
    def notes(values):
        for v in values:
            story.append(para("- " + str(v)))
    def table(rows, columns=None):
        if not rows:
            story.append(para("Not checked / Not provided."))
            return
        columns = columns or list(dict.fromkeys(k for r in rows for k in r))
        # Use multiple narrow tables rather than shrinking a 20-column analysis.
        for start in range(0, len(columns), 5):
            cols = columns[start:start+5]
            if start and columns[0] not in cols:
                cols = [columns[0]] + cols
            values = [[para(c.replace("_", " ").title(), "BusinessSmall") for c in cols]]
            for row in rows:
                def fmt(v):
                    if v is None:
                        return "Not available"
                    if isinstance(v, float):
                        return f"{v:.8g}"
                    if isinstance(v, (dict, list)):
                        return json.dumps(v, ensure_ascii=False)
                    return str(v)
                values.append([para(fmt(row.get(c)), "BusinessSmall") for c in cols])
            t = Table(values, colWidths=[174*mm/len(cols)]*len(cols), repeatRows=1, hAlign="LEFT")
            t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e9f2f4")),
                                   ("VALIGN", (0,0), (-1,-1), "TOP"), ("LINEBELOW", (0,0), (-1,0), .6, colors.HexColor("#b7ccd1")),
                                   ("LINEBELOW", (0,1), (-1,-1), .25, colors.HexColor("#e0e6eb")),
                                   ("LEFTPADDING", (0,0), (-1,-1), 6), ("RIGHTPADDING", (0,0), (-1,-1), 6),
                                   ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5)]))
            story.extend([t, Spacer(1, 5*mm)])
    def target_chart(rows):
        if not rows or len(rows)>10:
            return
        maximum=max(r["recruitment_n"] for r in rows)
        drawing=Drawing(480,35*len(rows)+22)
        for i,row in enumerate(rows):
            y=35*(len(rows)-i-1)+18
            name=str(row["arm"])
            drawing.add(String(0,y+6,name[:12]+("..." if len(name)>12 else ""),fontName=font,fontSize=9,fillColor=colors.HexColor("#24364a")))
            width=290*row["recruitment_n"]/maximum
            drawing.add(Rect(125,y,width,17,fillColor=colors.HexColor("#007f89"),strokeColor=None))
            drawing.add(String(132+width,y+5,str(row["recruitment_n"]),fontName=font,fontSize=8,fillColor=colors.HexColor("#24364a")))
        story.extend([drawing,para("Recruitment targets include the saved missing-outcome reserve.","BusinessSmall")])
    c = plan["config"]
    if c.get("synthetic"):
        story.append(para("SYNTHETIC DEMO - All inputs and outcomes are illustrative.","BusinessHeading"))
    story += [para("Experiment calculation memorandum", "BusinessTitle"), para(c["name"]),
              para(f"{stage} | Owner: {c['owner']} | Protocol v{plan['version']}"),
              para(f"Experiment ID: {c['experiment_id']}"),
              para(f"Protocol saved: {plan['saved_at']} | Snapshot calculated: {snapshot['calculated_at'] if snapshot else plan['saved_at']}"),
              para("Traceable calculation memorandum. This document does not establish completed audit or approval.")]
    heading("Decision and experiment definition")
    story.append(para(c["question"]))
    table(c["arms"], ["name", "role", "definition"])
    table([{k: c[k] for k in ("unit", "decision_scope", "start_date", "window_days", "maturity_delay_days", "eligible_per_week", "max_recruitment_weeks")}])
    heading("Metric definitions and decision thresholds")
    for m in c["metrics"]:
        story.append(para(f"{m['name']} - {m['role']} / {m['type']} / {m['direction']} is better"))
        table([m], ["name", "unit", "definition", "window_days", "effect", "effect_scale", "margin", "alert_threshold", "include_in_plan"])
    heading("Population and frozen weights")
    story.append(para(plan["estimator"]))
    table(plan["weights"])
    heading("Design assumptions")
    table([{k: c[k] for k in ("alpha", "power", "multiplicity", "attrition", "allocation", "assumption_only", "historical_start", "historical_end")}])
    story.append(para(f"Predeclared comparison family: {plan['family_size']}. Planning alpha per comparison: {plan['planning_alpha']:.10g}. Power is per planned comparison."))
    table(plan["variance_assumptions"])
    heading("Sample requirements and timing")
    target_chart(plan["arm_totals"])
    table(plan["arm_totals"])
    story.append(para(f"Recruitment duration: {plan['recruitment_weeks']:.6g} weeks. Recruitment end: {plan['recruitment_end']}. Earliest mature readout: {plan['earliest_readout']}. Within maximum recruitment duration: {plan['within_max_duration']}." if plan["targets"] else "Sample targets and recruitment calendar have not been calculated."))
    table(plan["targets"], ["partner", "segment", "arm", "analyzable_n", "recruitment_n", "expected_share"])
    heading("Pairwise Neyman and shared-control calculation")
    story.append(para("First solve each Test versus Control using the Primary metric SD ratio and all included metric constraints. Take the largest pairwise Control N. Fix it, solve each Test again, then verify all final integer stratum counts. Business-fixed allocation instead scales its declared ratios."))
    table(plan["pairwise"])
    table(plan["comparisons"], ["scope", "metric", "test", "control_n", "test_n", "detectable_change_or_margin", "planning_alpha", "power", "variance_of_difference", "degrees_of_freedom"])
    if snapshot:
        heading("Monitoring snapshot")
        story.append(para(f"Source: {snapshot['source']}. Data updated: {snapshot['updated_at']}. Downloading this memo does not refresh that timestamp."))
        table(snapshot["progress"])
        heading("Sample ratio mismatch")
        story.append(para(snapshot["srm"]["status"] + ": " + snapshot["srm"]["reason"]))
        table(snapshot["srm"]["rows"])
        heading("Pre-treatment balance")
        story.append(para(snapshot["balance"]["status"] + ": " + snapshot["balance"]["reason"]))
        table(snapshot["balance"]["rows"])
        heading("Guardrail monitoring and stratum coverage")
        table(snapshot.get("guardrails",[]))
        table(snapshot.get("cell_progress",[]),["partner","segment","arm","metric","analyzable_n","observed_n","target_met"])
        if snapshot.get("results"):
            heading("Effects, uncertainty and guardrails")
            story.append(para(snapshot["decision_status"]))
            notes(snapshot["blockers"])
            story.append(para(snapshot["multiplicity"]))
            for row in snapshot["results"]:
                story.append(para(f"{row['scope']} | {row['metric']} | {row['test']} vs {row['control']}", "BusinessHeading"))
                story.append(para(row["interpretation"]))
                table([row], ["metric", "control_n", "test_n", "control_mean", "test_mean", "effect", "relative_lift", "ci_lower", "ci_upper", "ci_level", "raw_p_value", "adjusted_p_value", "statistic", "degrees_of_freedom", "method", "guardrail_status", "guardrail_method"])
    heading("Assumptions, gaps and limitations")
    notes(plan["limitations"] + (snapshot.get("limitations", []) if snapshot else []))
    heading("Technical appendix: formulas and complete inputs")
    story.append(para("Tables display eight significant digits. Companion JSON retains full numeric precision for independent recalculation.","BusinessSmall"))
    notes(["Independent randomization units; mutually exclusive Partner x Segment strata; fixed weights identical in every arm.",
           "Estimate = sum_h w_h (mean_Test,h - mean_Control,h). Variance = sum_h w_h^2 (s_Test,h^2/n_Test,h + s_Control,h^2/n_Control,h).",
           "Welch-Satterthwaite df = variance^2 / sum_h,arm [(w_h^2 s_h,arm^2/n_h,arm)^2/(n_h,arm-1)].",
           "Continuous planning: noncentral t approximation with the same integer cell counts and df. Binary planning and large-cell testing: fixed-weight risk difference normal approximation; sparse cells require exact procedures.",
           "Continuous Guardrail: benefit-oriented difference compared with -margin; non-inferiority p-value is one-sided. Decisions use conservative family intervals, not p > 0.05.",
           "Adequate Binary cells use fixed-weight normal risk-difference intervals with family protection. Sparse cells use Newcombe-Wilson cell intervals with additional Bonferroni protection over strata; Fisher p-values do not invert these intervals.",
           "SMD uses pre-treatment covariates only. SRM uses assigned counts and frozen stratum allocation. Missing checks are never passes.",
           "Historic sample variance uses ddof=1 / VAR.S. Missing outcomes are not zeros; observed outcomes are not assignment counts."])
    table(plan["history"], ["partner", "segment", "metric", "arm", "sample_mean", "sample_variance", "sample_number", "successes"])
    if snapshot:
        table(snapshot["summaries"], ["partner", "segment", "metric", "arm", "sample_mean", "sample_variance", "sample_number", "successes"])
    heading("Reproduction fingerprints")
    if plan.get("runtime_versions"):
        table([plan["runtime_versions"]])
    for label, value in [("Code", plan["code_version"]), ("Configuration", plan["config_fingerprint"]), ("Historical inputs", plan["data_fingerprint"]), ("Snapshot inputs", snapshot["data_fingerprint"] if snapshot else "Not provided")]:
        story.append(para(f"{label}: {value}", "BusinessSmall"))
    story.append(para("Download the accompanying protocol JSON and aggregate CSV for machine-readable inputs. Individual customer IDs are excluded."))
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=19*mm, bottomMargin=19*mm,
                                 title=f"{c['name']} - {stage}", author=c["owner"])
    def footer(canvas, doc):
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.HexColor("#627487"))
        canvas.drawString(18*mm, 10*mm, f"Customer Business V2 | Protocol v{plan['version']}")
        canvas.drawRightString(192*mm, 10*mm, f"Page {doc.page}")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
