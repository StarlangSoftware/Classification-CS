using System.Collections.Generic;

namespace Classification.FeatureSelection
{
    public class ForwardSelection : SubSetSelection
    {
        /**
         * <summary> Constructor that creates a new {@link FeatureSubSet}.</summary>
         */
        public ForwardSelection() : base(new FeatureSubSet())
        {
        }

        /**
         * <summary> The operator method calls forward method which starts with having no feature in the model. In each iteration,
         * it keeps adding the features that are not currently listed.</summary>
         *
         * <param name="current">         FeatureSubset that will be added to new ArrayList.</param>
         * <param name="numberOfFeatures">Indicates the indices of indexList.</param>
         * <returns>ArrayList of FeatureSubSets created from forward.</returns>
         */
        protected override List<FeatureSubSet> Operator(FeatureSubSet current, int numberOfFeatures) {
            var result = new List<FeatureSubSet>();
            Forward(result, current, numberOfFeatures);
            return result;
        }
    }
}