namespace Classification.Model.DecisionTree
{
    public class DecisionStump : DecisionTree
    {
        /**
         * <summary> Training algorithm for C4.5 Stump univariate decision tree classifier.</summary>
         *
         * <param name="trainSet">  Training data given to the algorithm.</param>
         * <param name="parameters">-</param>
         */
        public override void Train(InstanceList.InstanceList trainSet, Parameter.Parameter parameters)
        {
            Root = new DecisionNode(trainSet, null, null, true);
        }

        /// <summary>
        /// Loads the decision tree model from an input file.
        /// </summary>
        /// <param name="fileName">File name of the decision tree model.</param>
        public override void LoadModel(string fileName)
        {
            Load(fileName);
        }

    }
}