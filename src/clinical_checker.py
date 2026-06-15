"""
clinical_checker.py — Clinical Decision Support (CDS) engine for cross-referencing prescriptions against lab results.
"""
from typing import List, Optional
from sqlalchemy.orm import Session
from src.database import get_latest_lab_results, ClinicalSafetyRule
from src.models import NormalizedDrug, ClinicalSafetyAlert

def check_clinical_safety(
    db: Session,
    patient_id: Optional[object],
    medications: List[NormalizedDrug]
) -> List[ClinicalSafetyAlert]:
    """
    Cross-references active prescription medications against the patient's latest lab results.
    Returns a list of high-priority clinician warning flags.
    """
    alerts: List[ClinicalSafetyAlert] = []
    if not patient_id:
        return alerts

    # Retrieve patient's latest unique diagnostic analytes
    latest_results = get_latest_lab_results(db, patient_id)
    if not latest_results:
        return alerts

    # Convert results list to a handy map for quick lookups: { "HBA1C": LabResultObj }
    lab_map = {res.analyte_name.upper(): res for res in latest_results}

    # Helper: check normalized active drugs in current prescription
    active_inns = {m.inn.lower() for m in medications if m.inn}

    # -----------------------------------------------------------------------
    # Step A: Query dynamic rules from the database
    # -----------------------------------------------------------------------
    dynamic_rules = []
    dynamically_checked = set()
    try:
        dynamic_rules = db.query(ClinicalSafetyRule).filter(
            ClinicalSafetyRule.is_enabled == True,
            ClinicalSafetyRule.is_deleted == False
        ).all()
    except Exception as e:
        pass

    for rule in dynamic_rules:
        drug_lower = rule.drug_inn.lower()
        analyte_upper = rule.analyte_name.upper()
        
        # Track this drug + analyte pair as dynamically handled
        dynamically_checked.add((drug_lower, analyte_upper))
        
        if drug_lower in active_inns:
            res = lab_map.get(analyte_upper)
            if res:
                # Heuristic gender check
                gender = rule.gender_specific or "both"
                if gender == "female":
                    # Female if "0.5" is in ref_range
                    if not (res.ref_range and "0.5" in res.ref_range):
                        continue
                elif gender == "male":
                    # Male if "0.5" is NOT in ref_range
                    if res.ref_range and "0.5" in res.ref_range:
                        continue
                        
                val = res.value
                matched = False
                
                # 1. Numeric Comparison
                if rule.operator:
                    op = rule.operator.strip()
                    thresh = rule.threshold_value
                    if op == ">" and thresh is not None and val > thresh:
                        matched = True
                    elif op == "<" and thresh is not None and val < thresh:
                        matched = True
                    elif op == ">=" and thresh is not None and val >= thresh:
                        matched = True
                    elif op == "<=" and thresh is not None and val <= thresh:
                        matched = True
                    elif op == "=" and thresh is not None and val == thresh:
                        matched = True
                    elif op == "between" and thresh is not None and rule.threshold_value_max is not None:
                        if thresh <= val <= rule.threshold_value_max:
                            matched = True
                # 2. Flag Comparison
                elif rule.flag_match:
                    if res.flag and res.flag.lower() == rule.flag_match.lower():
                        matched = True
                        
                if matched:
                    # Interpolate description template dynamically
                    desc = rule.description_template or ""
                    desc = desc.replace("{value}", f"{val:.2f}" if isinstance(val, float) else str(val))
                    desc = desc.replace("{unit}", res.unit or "")
                    desc = desc.replace("{threshold}", f"{rule.threshold_value:.2f}" if rule.threshold_value is not None else "")
                    
                    alerts.append(
                        ClinicalSafetyAlert(
                            drug_name=rule.drug_inn.title(),
                            analyte_name=rule.analyte_name.upper(),
                            severity=rule.severity or "warning",
                            description=desc,
                            val_detected=f"{val:.2f} {res.unit or ''}".strip(),
                            ref_range=res.ref_range or "",
                            management=rule.management_plan
                        )
                    )

    # Helper: Check if a fallback rule is allowed for a given drug + analyte
    def is_fallback_allowed(drug: str, analyte: str) -> bool:
        return (drug.lower(), analyte.upper()) not in dynamically_checked

    # -----------------------------------------------------------------------
    # Fallback Baseline Rules (Runs if no dynamic rule was checked for that pair)
    # -----------------------------------------------------------------------
    
    # Rule 1: Metformin + Elevated Serum Creatinine (Contraindication check)
    if "metformin" in active_inns and is_fallback_allowed("metformin", "CREATININE"):
        creat = lab_map.get("CREATININE")
        if creat:
            val = creat.value
            is_critical = False
            limit = 1.5
            
            # Simple heuristic check for gender/female reference range if present
            if creat.ref_range and "0.5" in creat.ref_range:
                limit = 1.4
                if val > 1.4:
                    is_critical = True
            else:
                if val > 1.5:
                    is_critical = True
                    
            if is_critical or val > 1.4:
                alerts.append(
                    ClinicalSafetyAlert(
                        drug_name="Metformin",
                        analyte_name="CREATININE",
                        severity="critical" if val > 1.5 else "warning",
                        description=(
                            f"Metformin is contraindicated due to elevated Serum Creatinine "
                            f"({val:.2f} {creat.unit or 'mg/dL'} > {limit} {creat.unit or 'mg/dL'} limit). "
                            f"Significantly elevated risk of Metformin-induced Lactic Acidosis."
                        ),
                        val_detected=f"{val:.2f} {creat.unit or 'mg/dL'}",
                        ref_range=creat.ref_range or "0.6 - 1.2 mg/dL",
                        management="Discontinue Metformin. Calculate eGFR and consider alternative glycemic agents."
                    )
                )

    # Rule 2: Levothyroxine + Out-of-Range TSH (Dose titration check)
    if "levothyroxine" in active_inns and is_fallback_allowed("levothyroxine", "TSH"):
        tsh = lab_map.get("TSH")
        if tsh:
            val = tsh.value
            if val > 4.5:
                alerts.append(
                    ClinicalSafetyAlert(
                        drug_name="Levothyroxine",
                        analyte_name="TSH",
                        severity="warning",
                        description=(
                            f"TSH level is elevated ({val:.2f} {tsh.unit or 'uIU/mL'} > 4.5 uIU/mL range), "
                            f"indicating potential under-dosing. Levothyroxine dose adjustment may be required."
                        ),
                        val_detected=f"{val:.2f} {tsh.unit or 'uIU/mL'}",
                        ref_range=tsh.ref_range or "0.4 - 4.5 uIU/mL",
                        management="Re-evaluate thyroid status and consider increasing Levothyroxine dose."
                    )
                )
            elif val < 0.4:
                alerts.append(
                    ClinicalSafetyAlert(
                        drug_name="Levothyroxine",
                        analyte_name="TSH",
                        severity="warning",
                        description=(
                            f"TSH level is suppressed ({val:.2f} {tsh.unit or 'uIU/mL'} < 0.4 uIU/mL range), "
                            f"indicating potential over-dosing. Risk of iatrogenic hyperthyroidism."
                        ),
                        val_detected=f"{val:.2f} {tsh.unit or 'uIU/mL'}",
                        ref_range=tsh.ref_range or "0.4 - 4.5 uIU/mL",
                        management="Re-evaluate thyroid status and consider reducing Levothyroxine dose."
                    )
                )

    # Rule 3: Aspirin + Suppressed Hemoglobin (Bleeding safety check)
    if "aspirin" in active_inns and is_fallback_allowed("aspirin", "HEMOGLOBIN"):
        hb = lab_map.get("HEMOGLOBIN")
        if hb:
            val = hb.value
            if val < 10.0:
                alerts.append(
                    ClinicalSafetyAlert(
                        drug_name="Aspirin",
                        analyte_name="HEMOGLOBIN",
                        severity="warning",
                        description=(
                            f"Severe anemia detected (Hemoglobin {val:.1f} {hb.unit or 'g/dL'} < 11.5 g/dL normal). "
                            f"Concomitant antiplatelet therapy (Aspirin) significantly elevates gastrointestinal bleeding risks."
                        ),
                        val_detected=f"{val:.1f} {hb.unit or 'g/dL'}",
                        ref_range=hb.ref_range or "12.0 - 16.0 g/dL",
                        management="Evaluate anemia source. Consider adding PPIs (e.g. Pantoprazole) for gastroprotection if Aspirin is mandatory."
                    )
                )

    # Rule 4: Statins + Elevated LDL Cholesterol (Efficacy assessment)
    statin_drugs = {
        "atorvastatin", "rosuvastatin", "simvastatin",
        "pravastatin", "pitavastatin", "fluvastatin", "lovastatin"
    }
    intersect_statins = active_inns.intersection(statin_drugs)
    if intersect_statins:
        statin_name = list(intersect_statins)[0].title()
        if is_fallback_allowed(statin_name.lower(), "LDL"):
            ldl = lab_map.get("LDL")
            if ldl:
                val = ldl.value
                if val > 100.0:
                    alerts.append(
                        ClinicalSafetyAlert(
                            drug_name=statin_name,
                            analyte_name="LDL",
                            severity="info",
                            description=(
                                f"LDL cholesterol is elevated ({val:.1f} {ldl.unit or 'mg/dL'} > 100 mg/dL target). "
                                f"Evaluating efficacy of active lipid-lowering therapy ({statin_name})."
                            ),
                            val_detected=f"{val:.1f} {ldl.unit or 'mg/dL'}",
                            ref_range=ldl.ref_range or "< 100 mg/dL",
                            management="Continue lipid-lowering statin therapy. Recheck lipid profile in 6-8 weeks."
                        )
                    )

    # Rule 5: Insulin or Sulfonylureas + Hypoglycemia
    hypo_drugs = {"insulin", "glimepiride", "gliclazide", "glipizide"}
    intersect_hypo = active_inns.intersection(hypo_drugs)
    if intersect_hypo:
        drug_name = list(intersect_hypo)[0].title()
        if is_fallback_allowed(drug_name.lower(), "FASTING_BLOOD_SUGAR"):
            fbs = lab_map.get("FASTING_BLOOD_SUGAR")
            if fbs:
                val = fbs.value
                if val < 70.0:
                    alerts.append(
                        ClinicalSafetyAlert(
                            drug_name=drug_name,
                            analyte_name="FASTING_BLOOD_SUGAR",
                            severity="critical",
                            description=(
                                f"Clinical hypoglycemia detected (Fasting Blood Sugar {val:.1f} {fbs.unit or 'mg/dL'} < 70 mg/dL). "
                                f"Concomitant hypoglycemic drug therapy ({drug_name}) poses severe risk of neuroglycopenia or coma."
                            ),
                            val_detected=f"{val:.1f} {fbs.unit or 'mg/dL'}",
                            ref_range=fbs.ref_range or "70 - 100 mg/dL",
                            management="Hold/reduce hypoglycemic drug dosage. Administer fast-acting glucose immediately and counsel patient on hypoglycemia protocols."
                        )
                    )

    # Rule 6: Antidiabetic Medications + Elevated HbA1c (Sub-optimal control)
    diabetic_drugs = {"metformin", "insulin", "glimepiride", "gliclazide", "sitagliptin", "empagliflozin", "pioglitazone"}
    intersect_diab = active_inns.intersection(diabetic_drugs)
    if intersect_diab:
        drug_name = list(intersect_diab)[0].title()
        if is_fallback_allowed(drug_name.lower(), "HBA1C"):
            hba1c = lab_map.get("HBA1C")
            if hba1c:
                val = hba1c.value
                if val > 8.0:
                    alerts.append(
                        ClinicalSafetyAlert(
                            drug_name=drug_name,
                            analyte_name="HBA1C",
                            severity="warning",
                            description=(
                                f"Poor glycemic control detected (HbA1c {val:.1f}% > 8.0%). "
                                f"Patient is under active pharmacotherapy ({drug_name}). Dose escalation or therapeutic intensification is indicated."
                            ),
                            val_detected=f"{val:.1f}%",
                            ref_range=hba1c.ref_range or "< 5.7 %",
                            management="Evaluate patient adherence, titrate active medications, or consider dual/triple combination oral therapy."
                        )
                    )

    # Rule 7: Antiplatelets + Suppressed Platelets (Thrombocytopenia check)
    antiplatelets = {"aspirin", "clopidogrel"}
    intersect_ap = active_inns.intersection(antiplatelets)
    if intersect_ap:
        drug_name = list(intersect_ap)[0].title()
        if is_fallback_allowed(drug_name.lower(), "PLATELET_COUNT"):
            plt = lab_map.get("PLATELET_COUNT")
            if plt:
                val = plt.value
                threshold = 100.0 if val < 1000.0 else 100000.0
                if val < threshold:
                    alerts.append(
                        ClinicalSafetyAlert(
                            drug_name=drug_name,
                            analyte_name="PLATELET_COUNT",
                            severity="critical",
                            description=(
                                f"Significant thrombocytopenia detected (Platelets {val:.1f} {plt.unit or 'x10^3/uL'}). "
                                f"Active antiplatelet therapy ({drug_name}) significantly increases severe hemorrhagic risks."
                            ),
                            val_detected=f"{val:.1f} {plt.unit or 'x10^3/uL'}",
                            ref_range=plt.ref_range or "150 - 450 x10^3/uL",
                            management="Hold antiplatelet agent. Investigate etiology of thrombocytopenia. Monitor patient for clinical bleeding signs."
                        )
                    )

    # Rule 8: ACE Inhibitors or ARBs + Elevated Serum Creatinine (AKI risk)
    ras_drugs = {"ramipril", "enalapril", "lisinopril", "telmisartan", "losartan", "valsartan"}
    intersect_ras = active_inns.intersection(ras_drugs)
    if intersect_ras:
        drug_name = list(intersect_ras)[0].title()
        if is_fallback_allowed(drug_name.lower(), "CREATININE"):
            creat = lab_map.get("CREATININE")
            if creat:
                val = creat.value
                if val > 1.4:
                    alerts.append(
                        ClinicalSafetyAlert(
                            drug_name=drug_name,
                            analyte_name="CREATININE",
                            severity="warning",
                            description=(
                                f"Concomitant RAS blocker ({drug_name}) with elevated Serum Creatinine ({val:.2f} {creat.unit or 'mg/dL'}). "
                                f"Risk of acute kidney injury (AKI) or severe hyperkalemia."
                            ),
                            val_detected=f"{val:.2f} {creat.unit or 'mg/dL'}",
                            ref_range=creat.ref_range or "0.6 - 1.2 mg/dL",
                            management="Monitor renal functions and serum potassium. Consider temporary discontinuation or dose reduction of RAS inhibitor."
                        )
                    )

    # Rule 9: Statins + Elevated Triglycerides (Persistent dyslipidemia)
    if intersect_statins:
        statin_name = list(intersect_statins)[0].title()
        if is_fallback_allowed(statin_name.lower(), "TRIGLYCERIDES"):
            tg = lab_map.get("TRIGLYCERIDES")
            if tg:
                val = tg.value
                if val > 200.0:
                    alerts.append(
                        ClinicalSafetyAlert(
                            drug_name=statin_name,
                            analyte_name="TRIGLYCERIDES",
                            severity="info",
                            description=(
                                f"Persistent hypertriglyceridemia detected ({val:.1f} {tg.unit or 'mg/dL'} > 150 mg/dL normal) "
                                f"under active lipid-lowering therapy ({statin_name})."
                            ),
                            val_detected=f"{val:.1f} {tg.unit or 'mg/dL'}",
                            ref_range=tg.ref_range or "< 150 mg/dL",
                            management="Evaluate dietary habits. Consider optimization of statin dose or adding Omega-3 fatty acids/Fibrates if clinically indicated."
                        )
                    )

    # Rule 10: Antiplatelets + Combined Bleeding Risk (Suppressed Hemoglobin)
    if intersect_ap:
        drug_name = list(intersect_ap)[0].title()
        if is_fallback_allowed(drug_name.lower(), "HEMOGLOBIN"):
            hb = lab_map.get("HEMOGLOBIN")
            if hb:
                val = hb.value
                if 10.0 <= val < 11.5:
                    alerts.append(
                        ClinicalSafetyAlert(
                            drug_name=drug_name,
                            analyte_name="HEMOGLOBIN",
                            severity="info",
                            description=(
                                f"Mild anemia detected (Hemoglobin {val:.1f} {hb.unit or 'g/dL'}). "
                                f"Evaluating bleeding risk for patient taking antiplatelet agent ({drug_name})."
                            ),
                            val_detected=f"{val:.1f} {hb.unit or 'g/dL'}",
                            ref_range=hb.ref_range or "12.0 - 16.0 g/dL",
                            management="Monitor complete blood count regularly. Assess patient for occult gastrointestinal blood loss."
                        )
                    )

    # Rule 11: Immunosuppressants / Cytotoxic + Low TLC or Low Neutrophils (Myelosuppression)
    cytotoxic_drugs = {"methotrexate", "azathioprine"}
    intersect_cyto = active_inns.intersection(cytotoxic_drugs)
    if intersect_cyto:
        drug_name = list(intersect_cyto)[0].title()
        tlc = lab_map.get("TLC")
        neut = lab_map.get("NEUTROPHILS")
        
        # Check Total Leucocyte Count (TLC) under 4.0 (x10^3/uL)
        if tlc and tlc.value < 4.0:
            if is_fallback_allowed(drug_name.lower(), "TLC"):
                val = tlc.value
                alerts.append(
                    ClinicalSafetyAlert(
                        drug_name=drug_name,
                        analyte_name="TLC",
                        severity="critical",
                        description=(
                            f"Leucopenia detected (Total Leucocyte Count {val:.2f} {tlc.unit or '10^3/uL'} < 4.0). "
                            f"Active myelosuppressive agent ({drug_name}) poses severe risk of neutropenic sepsis."
                        ),
                        val_detected=f"{val:.2f} {tlc.unit or '10^3/uL'}",
                        ref_range=tlc.ref_range or "4.0 - 11.0 10^3/uL",
                        management="Hold myelosuppressive agent. Perform urgent differential count. Counsel patient on immediately reporting fever/chills."
                    )
                )
        # Check Neutrophils percent under 40%
        elif neut and neut.value < 40.0:
            if is_fallback_allowed(drug_name.lower(), "NEUTROPHILS"):
                val = neut.value
                alerts.append(
                    ClinicalSafetyAlert(
                        drug_name=drug_name,
                        analyte_name="NEUTROPHILS",
                        severity="warning",
                        description=(
                            f"Relative neutropenia detected (Neutrophils {val:.1f}% < 40.0% range) "
                            f"with active myelosuppressive therapy ({drug_name})."
                        ),
                        val_detected=f"{val:.1f}%",
                        ref_range=neut.ref_range or "40.0 - 70.0 %",
                        management="Monitor absolute neutrophil count (ANC). Suspend myelosuppressive agent if ANC falls below 1500 cells/uL."
                    )
                )

    return alerts
