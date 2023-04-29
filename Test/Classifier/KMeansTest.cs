using Classification.Classifier;
using Classification.Parameter;
using NUnit.Framework;

namespace Test.Classifier
{
    public class KMeansTest : ClassifierTest
    {
        [Test]
        public void TestTrain()
        {
            var kMeans = new KMeans();
            var kMeansParameter = new KMeansParameter(1);
            kMeans.Train(iris.GetInstanceList(), kMeansParameter);
            Assert.AreEqual(7.33, 100 * kMeans.Test(iris.GetInstanceList()).GetErrorRate(), 0.01);
            kMeans.Train(bupa.GetInstanceList(), kMeansParameter);
            Assert.AreEqual(43.77, 100 * kMeans.Test(bupa.GetInstanceList()).GetErrorRate(), 0.01);
            kMeans.Train(dermatology.GetInstanceList(), kMeansParameter);
            Assert.AreEqual(45.08, 100 * kMeans.Test(dermatology.GetInstanceList()).GetErrorRate(), 0.01);
            kMeans.Train(car.GetInstanceList(), kMeansParameter);
            Assert.AreEqual(47.97, 100 * kMeans.Test(car.GetInstanceList()).GetErrorRate(), 0.01);
            kMeans.Train(tictactoe.GetInstanceList(), kMeansParameter);
            Assert.AreEqual(38.94, 100 * kMeans.Test(tictactoe.GetInstanceList()).GetErrorRate(), 0.01);
            kMeans.Train(nursery.GetInstanceList(), kMeansParameter);
            Assert.AreEqual(53.60, 100 * kMeans.Test(nursery.GetInstanceList()).GetErrorRate(), 0.01);
            kMeans.Train(chess.GetInstanceList(), kMeansParameter);
            Assert.AreEqual(83.25, 100 * kMeans.Test(chess.GetInstanceList()).GetErrorRate(), 0.01);
        }
        
        [Test]
        public void TestLoad()
        {
            var kMeans = new KMeans();
            kMeans.LoadModel("../../../models/kMeans-iris.txt");
            Assert.AreEqual(7.33, 100 * kMeans.Test(iris.GetInstanceList()).GetErrorRate(), 0.01);
            kMeans.LoadModel("../../../models/kMeans-bupa.txt");
            Assert.AreEqual(43.77, 100 * kMeans.Test(bupa.GetInstanceList()).GetErrorRate(), 0.01);
            kMeans.LoadModel("../../../models/kMeans-dermatology.txt");
            Assert.AreEqual(45.08, 100 * kMeans.Test(dermatology.GetInstanceList()).GetErrorRate(), 0.01);
            kMeans.LoadModel("../../../models/kMeans-car.txt");
            Assert.AreEqual(44.21, 100 * kMeans.Test(car.GetInstanceList()).GetErrorRate(), 0.01);
            kMeans.LoadModel("../../../models/kMeans-tictactoe.txt");
            Assert.AreEqual(38.94, 100 * kMeans.Test(tictactoe.GetInstanceList()).GetErrorRate(), 0.01);
            kMeans.LoadModel("../../../models/kMeans-nursery.txt");
            Assert.AreEqual(60.26, 100 * kMeans.Test(nursery.GetInstanceList()).GetErrorRate(), 0.01);
            kMeans.LoadModel("../../../models/kMeans-chess.txt");
            Assert.AreEqual(83.25, 100 * kMeans.Test(chess.GetInstanceList()).GetErrorRate(), 0.01);
        }

    }
}