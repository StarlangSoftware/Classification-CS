using System.Collections.Generic;

namespace Classification.FeatureSelection
{
    public class BackwardSelection : SubSetSelection
    {
        /**
         * <summary> Constructor that creates a new {@link FeatureSubSet} and initializes indexList with given number of features.</summary>
         *
         * <param name="numberOfFeatures">Indicates the indices of indexList.</param>
         */
        public BackwardSelection(int numberOfFeatures) : base(new FeatureSubSet(numberOfFeatures))
        {
        }

        /**
         * <summary> The operator method calls backward method which starts with all the features and removes the least significant feature at each iteration.</summary>
         *
         * <param name="current">         FeatureSubset that will be added to new ArrayList.</param>
         * <param name="numberOfFeatures">Indicates the indices of indexList.</param>
         * <returns>ArrayList of FeatureSubSets created from backward.</returns>
         */
        protected override List<FeatureSubSet> Operator(FeatureSubSet current, int numberOfFeatures) {
            var result = new List<FeatureSubSet>();
            Backward(result, current);
            return result;
        }
    }
}