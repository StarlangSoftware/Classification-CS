using Classification.Model;
using Classification.Parameter;
using NUnit.Framework;

namespace Test.Classifier
{
    public class LinearPerceptronTest : ClassifierTest
    {
        [Test]
        public void TestTrain()
        {
            var linearPerceptron = new LinearPerceptronModel();
            var linearPerceptronParameter = new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100);
            linearPerceptron.Train(iris.GetInstanceList(), linearPerceptronParameter);
            Assert.AreEqual(2.67, 100 * linearPerceptron.Test(iris.GetInstanceList()).GetErrorRate(), 0.01);
            linearPerceptronParameter = new LinearPerceptronParameter(1, 0.001, 0.99, 0.2, 100);
            linearPerceptron.Train(bupa.GetInstanceList(), linearPerceptronParameter);
            Assert.AreEqual(29.28, 100 * linearPerceptron.Test(bupa.GetInstanceList()).GetErrorRate(), 0.01);
            linearPerceptronParameter = new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100);
            linearPerceptron.Train(dermatology.GetInstanceList(), linearPerceptronParameter);
            Assert.AreEqual(3.55, 100 * linearPerceptron.Test(dermatology.GetInstanceList()).GetErrorRate(), 0.01);
        }
        
        [Test]
        public void TestLoad()
        {
            var linearPerceptron = new LinearPerceptronModel();
            linearPerceptron.LoadModel("../../../models/linearPerceptron-iris.txt");
            Assert.AreEqual(3.33, 100 * linearPerceptron.Test(iris.GetInstanceList()).GetErrorRate(), 0.01);
            linearPerceptron.LoadModel("../../../models/linearPerceptron-bupa.txt");
            Assert.AreEqual(31.88, 100 * linearPerceptron.Test(bupa.GetInstanceList()).GetErrorRate(), 0.01);
            linearPerceptron.LoadModel("../../../models/linearPerceptron-dermatology.txt");
            Assert.AreEqual(0.82, 100 * linearPerceptron.Test(dermatology.GetInstanceList()).GetErrorRate(), 0.01);
        }

    }
}