import pickle
import inflection
import pandas as pd
import numpy  as np

class Census( object ):
    
    def __init__( self ):
        self.home_path = ''

        self.age_scaler             = pickle.load( open( self.home_path +'parameters/age_scaler.pkl', 'rb' ) )
        self.education_num_scaler   = pickle.load( open( self.home_path +'parameters/education_num_scaler.pkl', 'rb' ) )
        self.final_weight_scaler    = pickle.load( open( self.home_path +'parameters/final_weight_scaler.pkl', 'rb' ) )
        self.hour_per_week_scaler   = pickle.load( open( self.home_path +'parameters/hour_per_week_scaler.pkl', 'rb' ) )
        self.sex_encoding           = pickle.load( open( self.home_path +'parameters/sex_encoding.pkl', 'rb' ) )
        self.relationship_encoding  = pickle.load( open( self.home_path +'parameters/relationship_encoding.pkl', 'rb' ) )

        
    def data_selection( self, df1 ):
        old_columns = ['age', 'workclass', 'final_weight', 'education', 'education_num',
               'marital_status', 'occupation', 'relationship', 'race', 'sex',
               'capital_gain', 'capital_loos', 'hour_per_week', 'native_country']

        snakecase = lambda x: inflection.underscore( x )
        new_cols = list( map( snakecase, old_columns ) )

        df1.columns = new_cols
        
        df1 = df1.drop( ['education', 'capital_gain', 'capital_loos', 'native_country'], axis=1 )
        
        return df1
    
    def data_preparation( self, df5 ):

        ###### Rescaling ######
        
        # MinMaxScaler
        df5['age']           = self.age_scaler.transform( df5[['age']].values )

        df5['education_num'] = self.education_num_scaler.transform( df5[['education_num']].values )

        # RobustScaler
        df5['final_weight']  = self.final_weight_scaler.transform( df5[['final_weight']].values )

        df5['hour_per_week'] = self.hour_per_week_scaler.transform( df5[['hour_per_week']].values )
        
        
        ###### dictionary to encoding ######

        dic_race = {' White': 0.7383396432613871,
         ' Black': 0.8760593220338984,
         ' Amer-Indian-Eskimo': 0.8817567567567568,
         ' Asian-Pac-Islander': 0.7671232876712328,
         ' Other': 0.8992248062015504}

        dic_marital_status = {' Married-civ-spouse': 0.5424147217235189,
         ' Never-married': 0.9532310262031527,
         ' Divorced': 0.8954829408938011,
         ' Widowed': 0.9168514412416852,
         ' Separated': 0.9320498301245753,
         ' Married-spouse-absent': 0.8972332015810277,
         ' Married-AF-spouse': 0.5652173913043478}

        dic_workclass = {
            ' Private': 0.774472,
            ' Self-emp-not-inc': 0.715089,
            ' Local-gov': 0.706033,
            ' ?': 0.893912,
            ' State-gov': 0.730579,
            ' Self-emp-inc': 0.436932,
            ' Federal-gov': 0.620767,
            ' Without-pay': 1.000000,
            ' Never-worked': 1.000000}

        dic_occupation = {' Exec-managerial': 0.5132530120481927,
         ' Prof-specialty': 0.553208773354996,
         ' Craft-repair': 0.7666214382632293,
         ' Adm-clerical': 0.8672078863438678,
         ' Sales': 0.7241379310344828,
         ' Other-service': 0.9618293122074181,
         ' Machine-op-inspct': 0.8672199170124482,
         ' ?': 0.8943577430972389,
         ' Transport-moving': 0.795439302481556,
         ' Handlers-cleaners': 0.9386038687973086,
         ' Farming-fishing': 0.8737201365187713,
         ' Tech-support': 0.6976470588235294,
         ' Protective-serv': 0.665016501650165,
         ' Priv-house-serv': 0.9888888888888889,
         ' Armed-Forces': 0.8888888888888888}
        
        ###### Encoding ######
        
        ####### label Encoding ######
        # sex
        df5['sex']    = self.sex_encoding.transform( df5['sex'] )
        
        ####### OneHotEncoding #########

        # relationship
        c = self.relationship_encoding.transform( df5[['relationship']] ).toarray()
        c = pd.DataFrame( c )
        df5 = pd.concat( [df5, c], axis=1 ).drop( ['relationship'], axis=1 )

        ####### Target Encoding #########

        # occupation
        df5['occupation']     = df5['occupation'].map( dic_occupation )

        # workclass
        df5['workclass']      = df5['workclass'] .map( dic_workclass )

        # marital_status
        df5['marital_status'] = df5['marital_status'].map( dic_marital_status )

        # race
        df5['race']           = df5['race'].map( dic_race )
        
        return df5
    
    def get_predictions( self, model, original_data, test_data):
        pred = model.predict( test_data )
        
        original_data['prediction'] = pd.Series( pred ).map( {1: " <=50K", 0: " >50K"} )
        
        return original_data.to_json( orient='records', date_format='iso' )