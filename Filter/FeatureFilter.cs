namespace Classification.Filter
{
    public abstract class FeatureFilter
    {
        protected DataSet.DataSet dataSet;

        protected abstract void ConvertInstance(Instance.Instance instance);

        protected abstract void ConvertDataDefinition();

        /**
         * <summary> Constructor that sets the dataSet.</summary>
         *
         * <param name="dataSet">DataSet that will bu used.</param>
         */
        public FeatureFilter(DataSet.DataSet dataSet)
        {
            this.dataSet = dataSet;
        }

        /**
         * <summary> Feature converter for a list of instances. Using the abstract method convertInstance, each instance in the
         * instance list will be converted.</summary>
         */
        public void Convert()
        {
            var instances = dataSet.GetInstances();
            foreach (var instance in instances) {
                ConvertInstance(instance);
            }
            ConvertDataDefinition();
        }
    }
}