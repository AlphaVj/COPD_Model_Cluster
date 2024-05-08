from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .utils import Copd_predict_output


class PredictCOPDView(APIView):
    def post(self, request):
        # Get input data from the request
        input_data = request.data
        
        # Extract values from the input data dictionary and convert to a single list
        features = [
            input_data["Age"],
            input_data["PackHistory"],
            input_data["MWT1Best"],
            input_data["FEV1"],
            input_data["FEV1PRED"],
            input_data["FVC"],
            input_data["FVCPRED"],
            input_data["CAT"],
            input_data["HAD"],
            input_data["SGRQ"],
            input_data["AGEquartiles"],
            input_data["gender"],
            input_data["smoking"],
            input_data["Diabetes"],
            input_data["muscular"],
            input_data["hypertension"],
            input_data["AtrialFib"],
            input_data["IHD"]
        ]
        
        # Call the predict_output function with the input data
        predicted_label = Copd_predict_output(features)
        
        # Return the predicted label in the response
        return Response({"predicted_label": predicted_label}, status=status.HTTP_200_OK)



