namespace Classification.InstanceList
{
    public class InstanceListOfSameClass : InstanceList
    {
        private readonly string _classLabel;

        /**
         * <summary> Constructor for creating a new instance list with the same class label.</summary>
         *
         * <param name="classLabel">Class label of instance list.</param>
         */
        public InstanceListOfSameClass(string classLabel)
        {
            this._classLabel = classLabel;
        }

        /**
         * <summary> Accessor for the class label.</summary>
         *
         * <returns>Class label.</returns>
         */
        public string GetClassLabel()
        {
            return _classLabel;
        }
    }
}