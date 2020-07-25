using System.Collections.Generic;

namespace Classification.FeatureSelection
{
    public class FloatingSelection : SubSetSelection
    {
        /**
         * <summary> Constructor that creates a new {@link FeatureSubSet}.</summary>
         */
        public FloatingSelection() : base(new FeatureSubSet())
        {
        }

        /**
         * <summary> The operator method calls forward and backward methods.</summary>
         *
         * <param name="current">         {@link FeatureSubSet} input.</param>
         * <param name="numberOfFeatures">Indicates the indices of indexList.</param>
         * <returns>ArrayList of FeatureSubSet.</returns>
         */
        protected override List<FeatureSubSet> Operator(FeatureSubSet current, int numberOfFeatures) {
            var result = new List<FeatureSubSet>();
            Forward(result, current, numberOfFeatures);
            Backward(result, current);
            return result;
        }
    }
}