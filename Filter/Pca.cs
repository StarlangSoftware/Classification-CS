using System.Collections.Generic;
using Classification.Attribute;
using Math;

namespace Classification.Filter
{
    public class Pca : TrainedFeatureFilter
    {
        private readonly double _covarianceExplained = 0.99;
        private List<Eigenvector> _eigenvectors;
        private readonly int _numberOfDimensions = -1;

        /**
         * <summary> Constructor that sets the dataSet and covariance explained. Then calls train method.</summary>
         *
         * <param name="dataSet">            DataSet that will bu used.</param>
         * <param name="covarianceExplained">Number that shows the explained covariance.</param>
         */
        public Pca(DataSet.DataSet dataSet, double covarianceExplained) : base(dataSet)
        {
            this._covarianceExplained = covarianceExplained;
            Train();
        }

        /**
         * <summary> Constructor that sets the dataSet and dimension. Then calls train method.</summary>
         *
         * <param name="dataSet">           DataSet that will bu used.</param>
         * <param name="numberOfDimensions">Dimension number.</param>
         */
        public Pca(DataSet.DataSet dataSet, int numberOfDimensions) : base(dataSet)
        {
            this._numberOfDimensions = numberOfDimensions;
            Train();
        }

        /**
         * <summary> Constructor that sets the dataSet and dimension. Then calls train method.</summary>
         *
         * <param name="dataSet">DataSet that will bu used.</param>
         */
        public Pca(DataSet.DataSet dataSet) : base(dataSet)
        {
            Train();
        }

        /**
         * <summary> The removeUnnecessaryEigenvectors methods takes an ArrayList of Eigenvectors. It first calculates the summation
         * of eigenValues. Then it finds the eigenvectors which have lesser summation than covarianceExplained and removes these
         * eigenvectors.</summary>
         */
        private void RemoveUnnecessaryEigenvectors()
        {
            double sum = 0.0, currentSum = 0.0;
            foreach (var eigenvector in _eigenvectors)
            {
                sum += eigenvector.GetEigenValue();
            }

            for (var i = 0; i < _eigenvectors.Count; i++)
            {
                if (currentSum / sum < _covarianceExplained)
                {
                    currentSum += _eigenvectors[i].GetEigenValue();
                }
                else
                {
                    _eigenvectors = _eigenvectors.GetRange(i, _eigenvectors.Count - i);
                    break;
                }
            }
        }

        /**
         * <summary> The removeAllEigenvectorsExceptTheMostImportantK method takes an {@link ArrayList} of {@link Eigenvector}s and removes the
         * surplus eigenvectors when the number of eigenvectors is greater than the dimension.</summary>
         */
        private void RemoveAllEigenvectorsExceptTheMostImportantK()
        {
            _eigenvectors = _eigenvectors.GetRange(0, _numberOfDimensions);
        }

        /**
         * <summary> The train method creates an averageVector from continuousAttributeAverage and a covariance {@link Matrix} from that averageVector
         * Then finds the eigenvectors of that covariance matrix and removes its unnecessary eigenvectors..</summary>
         */
        protected sealed override void Train()
        {
            var averageVector = new Vector(dataSet.GetInstanceList().ContinuousAttributeAverage());
            var covariance = dataSet.GetInstanceList().Covariance(averageVector);
            _eigenvectors = covariance.Characteristics();
            if (_numberOfDimensions != -1)
            {
                RemoveAllEigenvectorsExceptTheMostImportantK();
            }
            else
            {
                RemoveUnnecessaryEigenvectors();
            }
        }

        /**
         * <summary> The convertInstance method takes an {@link Instance} as an input and creates a {@link java.util.Vector} attributes from continuousAttributes.
         * After removing all attributes of given instance, it then adds new {@link ContinuousAttribute} by using the dot
         * product of attributes Vector and the eigenvectors.</summary>
         *
         * <param name="instance">Instance that will be converted to {@link ContinuousAttribute} by using eigenvectors.</param>
         */
        protected override void ConvertInstance(Instance.Instance instance)
        {
            var attributes = new Vector(instance.ContinuousAttributes());
            instance.RemoveAllAttributes();
            foreach (var eigenvector in _eigenvectors)
            {
                instance.AddAttribute(new ContinuousAttribute(attributes.DotProduct(eigenvector)));
            }
        }

        /**
         * <summary> The convertDataDefinition method gets the data definitions of the dataSet and removes all the attributes. Then adds
         * new attributes as CONTINUOUS.</summary>
         */
        protected override void ConvertDataDefinition()
        {
            var dataDefinition = dataSet.GetDataDefinition();
            dataDefinition.RemoveAllAttributes();
            for (int i = 0; i < _eigenvectors.Count; i++)
            {
                dataDefinition.AddAttribute(AttributeType.CONTINUOUS);
            }
        }
    }
}