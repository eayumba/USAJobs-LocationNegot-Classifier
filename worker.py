import pandas as pd


from jobs_classifier.text_preprocessing.preprocessing import Preprocess
from jobs_classifier import model_setup
from jobs_classifier import enums


def inference(text, print_out=True):
    if type(text) == str:
        text = [text]

    clean_text = [Preprocess(t).text.lower() for t in text]

    tfidf_module = model_setup.tfidf_trainer(pretrained_model=True)
    tfidf_out = tfidf_module.transform(clean_text)

    model = model_setup.JobSummaryClassifier(load_pretrained=True)
    predictions = model.predict(tfidf_out)[:, 1]

    output_summary = pd.DataFrame({"Text": text, "Confidence": list(predictions)})
    output_summary["Class"] = output_summary["Confidence"] >= enums.THRESHOLD

    output_summary.to_csv("model_predictions/inference.csv", index=False)
    
    
    if print_out:
        for ix, (t, pred) in enumerate(zip(text, predictions)):
    
            if pred >= enums.THRESHOLD:
                if pred >= enums.HIGH_CONFIDENCE:
                    print(f"\n{t} -------------> \n [GREEN]: Location Negotiable EGLIBLE with high confidence")
                else: print(f"\n{t} -------------> \n [YELLOW]: Location Negotiable EGLIBLE medium confidence")
            
            else:
                if 1-pred >= enums.HIGH_CONFIDENCE:
                    print(f"\n{t} -------------> \n [RED]: Location Negotiable INEGLIBLE with high confidence")
                else: print(f"\n{t} -------------> \n [YELLOW]: Location Negotiable INEGLIBLE with medium confidence")


if __name__ == "__main__":
    remote_job_summary = """
    FBI Special Agents apply their professional expertise and unique skill sets to their work every day and that 
    includes law enforcement and military backgrounds. We're currently seeking police officers, military 
    veterans (Special Forces, explosives, WMD, intelligence) and pilots (helicopter and fixed-wing). 
    Apply your tactical skills, leadership, integrity and teamwork to gathering evidence or helping to 
    # dismantle a criminal enterprise.Special Agent - Law Enforcement or Military Veteran BackgroundCriminal Investigation
    """

    non_remote_job_summary = """
    Transportation Security Officers are responsible for providing security and protection of travelers across all 
    transportation sectors in a courteous and professional manner. Their duties may also extend to securing high-profile events, 
    important figures and/or anything that includes or impacts our transportation systems. 
    Learn more about the Transportation Security Officer (TSO) role on the TSA Careers Website.
    Transportation Security Officer (TSO)Compliance Inspection And Support
    
    """

    new_remote_job_summary = """
    The VISN 10 Clinical Resource Hub is recruiting for six Physicians (Primary Care). The Physicians (Primary Care) will
    function virtually within the VISN 10 Clinical Resource Hub, a repository of clinical &amp; administrative staff that 
    serve VA facilities within VISN 10 that are underserved or experiencing gaps in clinicians due to the inability to match 
    provider supply with demand. Services provided by CRH staff are conducted via multiple modalities, including virtual 
    and in-person care. This is a public notice of upcoming announcement for FAA authorized Computer Engineer, Electronic 
    Engineer, Computer Scientist, Information Technology Specialist Cybersecurity positions open to U.S. Citizens. Positions 
    may be announced for FV-H through K or grades FG-12 through 15. This position is located in the Health Benefits, 
    Memorial, Corporate (HBMC) organization, Office of Information and Technology (OI&T) serving as an Senior IT Specialist 
    responsible for the full range of duties associated with one or more IT specialty areas required to support: 
    FIELD DEVELOPMENT. This is a 100% Remote Work position. We are seeking a motivated, customer service oriented 
    professional to serve as Contract Specialist, with occasional travel to Arlington, VA and Paris. You will report to 
    the Director of Contracting, located in Paris. The first review of applications will be after 30 September 2021. 
    This announcement will remain open until the position is filled.
    """

    for posting in [remote_job_summary, non_remote_job_summary, new_remote_job_summary]:
        inference(text=posting)
