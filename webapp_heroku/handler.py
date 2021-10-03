import os
import pickle
import pandas as pd
from flask         import Flask, request, Response
from census.Census import Census

model = pickle.load( open( 'model/random_forest_tuned.pkl', 'rb' ) )

app = Flask( __name__ )
@app.route( '/census/predict', methods=['POST'] )

def census_predict():
    test_json = request.get_json()
    
    if test_json:
        if isinstance( test_json, dict ):
            test_raw = pd.DataFrame( test_json, index=[0] )
        else:
            test_raw = pd.DataFrame( test_json, columns=test_json[0].keys() )
        
        # Instantiate Census Class
        pipeline = Census()
        
        # data selection
        df1 = pipeline.data_selection( test_raw )
        
        # data preparation
        df2 = pipeline.data_preparation( df1 )
        
        # data prediction
        df_response = pipeline.get_predictions( model, test_raw, df2 )
        
        return df_response
    else:
        return Response( '{}', status=200, mimetype='application/json' )
    
if __name__ == '__main__':
    port = os.environ.get( 'PORT', 5000)
    app.run( '0.0.0.0', port=port )