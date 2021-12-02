import requests
import numpy as np
import pandas as pd

import local_config
import enums
import feature_engine as fe


class ParseOuput:
    def object_descriptors(self, data_object, descriptor_fields):
        """
        Tabluation of the base despcriptor fields into
        a pandas dataframe
        """
        base_storage = {enums.MOI: []}
        for page in data_object:
            for page_detail in page:
                base_storage[enums.MOI].append(page_detail[enums.MOI])
                descriptors = page_detail[enums.MOD]
                for base_fields in descriptor_fields:
                    base_storage.setdefault(base_fields, [])
                    if base_fields in descriptors:
                        base_storage[base_fields].append(descriptors[base_fields])
                    else:
                        base_storage[base_fields].append(None)
        return pd.DataFrame.from_dict(base_storage)


class ExtractAPIData(ParseOuput):
    def __init__(
        self, user: str, apikey: str, page_nums: int, result_limit: int, keyword: str
    ):
        self.user = user
        self.apikey = apikey
        self.page_nums = page_nums
        self.result_limit = result_limit
        self.keyword = keyword
        self.json_results = [self.extract(page) for page in range(self.page_nums)]
        self.base_results_df = self.object_descriptors(
            self.json_results, descriptor_fields=enums.BASE_DESCRIPTOR_FIELDS
        )
        self.extended_results_df = self.object_descriptors(
            self.json_results, descriptor_fields=enums.EXTENDED_DESCRIPTOR_FIELDS
        )

    def extract(self, page):

        headers = {
            "Host": "data.usajobs.gov",
            "User-Agent": self.user,
            "Authorization-Key": self.apikey,
        }

        # base_url = f"https://data.usajobs.gov/api/search?Page={page}&ResultsPerPage={self.result_limit}&SearchResult={self.keyword}"
        base_url = f"https://data.usajobs.gov/api/search?Page={page}&ResultsPerPage={self.result_limit}"
        url = base_url
        response = requests.get(url, headers=headers)
        data = response.json()
        return data["SearchResult"]["SearchResultItems"]
