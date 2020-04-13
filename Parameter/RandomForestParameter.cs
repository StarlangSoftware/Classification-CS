namespace Classification.Parameter
{
    public class RandomForestParameter : BaggingParameter
    {
        private readonly int _attributeSubsetSize;

        /**
         * <summary> Parameters of the random forest classifier.</summary>
         *
         * <param name="seed">               Seed is used for random number generation.</param>
         * <param name="ensembleSize">       The number of trees in the bagged forest.</param>
         * <param name="attributeSubsetSize">Integer value for the size of attribute subset.</param>
         */
        public RandomForestParameter(int seed, int ensembleSize, int attributeSubsetSize) : base(seed, ensembleSize)
        {
            this._attributeSubsetSize = attributeSubsetSize;
        }

        /**
         * <summary> Accessor for the attributeSubsetSize.</summary>
         *
         * <returns>The attributeSubsetSize.</returns>
         */
        public int GetAttributeSubsetSize()
        {
            return _attributeSubsetSize;
        }
    }
}