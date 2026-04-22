import random

class DiseaseClassifier:

    def predict_from_crops(self, crops):

        results = {}

        for region in crops:

            results[region] = random.uniform(0.02,0.15)

        return results