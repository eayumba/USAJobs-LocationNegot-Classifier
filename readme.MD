## USA Jobs Classifier

USA Jobs Search Information: https://developer.usajobs.gov/API-Reference/GET-api-Search

## Summary of Remote Jobs Classifier
From the USA Jobs API, we're extracting `8375` unique `Job Summary`, `Job Title`, and `Job Category` combinations. We did not use any keyword search terms in our API call as this may bias the results. The observed distribution of remote eligible jobs is rougly `2.6% (~217 jobs)`. Remote-work eligibility was determined by finding `PositionLocationDisplay's` containing `"Location Negotiable"`.

Key features used to build our model are in the table below. We concatenate the three fields into a single text field which is then fed into a stacked `tfidf-charwb-level` and `tfidf-word-level` [embedding model](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html). 

| Field Name |
| :-----:  |
| JobSummary |
| PositionTitle |
| JobCategory |

### Model Performance

| Model | Precision  | Recall  | F1 |
| :---:   | :-: | :-: | :-: |
| char_wb tfidf+word tfidf + SVM(linear kernel) | .92 | .54 | .68 |

Because `False Positives` are more expensive than `False Negatives`, we tune towards `precision` using a `.58` threshold cutoff point for classifying a job listing as `Remote Eligible` vs. `Non-Remote Eligible`.


### Demonstration of the `worker.py`
The worker currently can ingest a singular or a list of text values. Although not a requirement, the input text should be a concatenation of the three key fields listed above. Our model will score any text value fed into the inference layer, but the best performance will be yielded if provided the job summary, title, and category. 

![image](https://user-images.githubusercontent.com/94016666/142894285-7e135131-6ef3-48a7-b150-f67e10fee4fe.png)