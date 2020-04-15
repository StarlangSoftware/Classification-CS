using Classification.Model.DecisionTree;

namespace Classification.Classifier
{
    public class C45Stump : Classifier
    {
        /**
         * <summary> Training algorithm for C4.5 Stump univariate decision tree classifier.</summary>
         *
         * <param name="trainSet">  Training data given to the algorithm.</param>
         * <param name="parameters">-</param>
         */
        public override void Train(InstanceList.InstanceList trainSet, Parameter.Parameter parameters)
        {
            model = new DecisionTree(new DecisionNode(trainSet, null, null, true));
        }
    }
}