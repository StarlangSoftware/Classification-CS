namespace Classification.Experiment
{
    public class Experiment
    {
        private readonly Classifier.Classifier _classifier;
        private readonly Parameter.Parameter _parameter;
        private readonly DataSet.DataSet _dataSet;

        /**
         * <summary> Constructor for a specific machine learning experiment</summary>
         * <param name="classifier">Classifier used in the machine learning experiment</param>
         * <param name="parameter">Parameter(s) of the classifier.</param>
         * <param name="dataSet">DataSet on which the classifier is run.</param>
         */
        public Experiment(Classifier.Classifier classifier, Parameter.Parameter parameter, DataSet.DataSet dataSet)
        {
            this._classifier = classifier;
            this._parameter = parameter;
            this._dataSet = dataSet;
        }

        /**
         * <summary> Accessor for the classifier attribute.</summary>
         * <returns>Classifier attribute.</returns>
         */
        public Classifier.Classifier GetClassifier()
        {
            return _classifier;
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
    }
}