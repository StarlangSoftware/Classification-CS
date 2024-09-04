using Classification.FeatureSelection;

namespace Classification.Experiment
{
    public class Experiment
    {
        private readonly Model.Model _model;
        private readonly Parameter.Parameter _parameter;
        private readonly DataSet.DataSet _dataSet;

        /**
         * <summary> Constructor for a specific machine learning experiment</summary>
         * <param name="model">Classifier used in the machine learning experiment</param>
         * <param name="parameter">Parameter(s) of the model.</param>
         * <param name="dataSet">DataSet on which the model is run.</param>
         */
        public Experiment(Model.Model model, Parameter.Parameter parameter, DataSet.DataSet dataSet)
        {
            this._model = model;
            this._parameter = parameter;
            this._dataSet = dataSet;
        }

        /**
         * <summary> Accessor for the classifier attribute.</summary>
         * <returns>Classifier attribute.</returns>
         */
        public Model.Model GetModel()
        {
            return _model;
        }

        /**
         * <summary> Accessor for the parameter attribute.</summary>
         * <returns>Parameter attribute.</returns>
         */
        public Parameter.Parameter GetParameter()
        {
            return _parameter;
        }

        /**
         * <summary> Accessor for the dataSet attribute.</summary>
         * <returns>DataSet attribute.</returns>
         */
        public DataSet.DataSet GetDataSet()
        {
            return _dataSet;
        }
        
        /**
         * <summary>Construct and returns a feature selection experiment.</summary>
         * <param name="featureSubSet">Feature subset used in the feature selection experiment</param>
         * <returns>Experiment constructed</returns>
         */
        public Experiment FeatureSelectedExperiment(FeatureSubSet featureSubSet) {
            return new Experiment(_model, _parameter, _dataSet.GetSubSetOfFeatures(featureSubSet));
        }

    }
}