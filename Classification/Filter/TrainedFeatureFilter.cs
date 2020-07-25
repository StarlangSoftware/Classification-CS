namespace Classification.Filter
{
    public abstract class TrainedFeatureFilter : FeatureFilter
    {
        protected abstract void Train();

        /**
         * <summary> Constructor that sets the dataSet.</summary>
         *
         * <param name="dataSet">DataSet that will bu used.</param>
         */
        public TrainedFeatureFilter(DataSet.DataSet dataSet) : base(dataSet)
        {
        }
    }
}