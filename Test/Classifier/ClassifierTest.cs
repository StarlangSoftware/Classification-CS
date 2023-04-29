using System.Collections.Generic;
using Classification.Attribute;
using Classification.DataSet;
using NUnit.Framework;

namespace Test.Classifier
{
    public class ClassifierTest
    {
        protected Classification.DataSet.DataSet iris, car, chess, bupa, tictactoe, dermatology, nursery, carIndexed, tictactoeIndexed;

        [SetUp]
        public void SetUp()
        {
            var attributeTypes = new List<AttributeType>();

            for (var i = 0; i < 4; i++)
            {
                attributeTypes.Add(AttributeType.CONTINUOUS);
            }

            var dataDefinition = new DataDefinition(attributeTypes);
            iris = new Classification.DataSet.DataSet(dataDefinition, ",", "../../../datasets/iris.data");
            attributeTypes = new List<AttributeType>();
            for (var i = 0; i < 6; i++)
            {
                attributeTypes.Add(AttributeType.CONTINUOUS);
            }

            dataDefinition = new DataDefinition(attributeTypes);
            bupa = new Classification.DataSet.DataSet(dataDefinition, ",", "../../../datasets/bupa.data");
            attributeTypes = new List<AttributeType>();
            for (var i = 0; i < 34; i++)
            {
                attributeTypes.Add(AttributeType.CONTINUOUS);
            }

            dataDefinition = new DataDefinition(attributeTypes);
            dermatology = new Classification.DataSet.DataSet(dataDefinition, ",", "../../../datasets/dermatology.data");
            attributeTypes = new List<AttributeType>();
            for (var i = 0; i < 6; i++)
            {
                attributeTypes.Add(AttributeType.DISCRETE);
            }

            dataDefinition = new DataDefinition(attributeTypes);
            car = new Classification.DataSet.DataSet(dataDefinition, ",", "../../../datasets/car.data");
            attributeTypes = new List<AttributeType>();
            var attributeValueList = new string[6][];
            attributeValueList[0] = new [] { "vhigh", "high", "low", "med" };
            attributeValueList[1] = new [] { "vhigh", "high", "low", "med" };
            attributeValueList[2] = new [] { "2", "3", "4", "5more" };
            attributeValueList[3] = new [] { "2", "4", "more" };
            attributeValueList[4] = new [] { "big", "med", "small" };
            attributeValueList[5] = new [] { "high", "low", "med" };
            for (var i = 0; i < 6; i++)
            {
                attributeTypes.Add(AttributeType.DISCRETE_INDEXED);
            }

            dataDefinition = new DataDefinition(attributeTypes, attributeValueList);
            carIndexed = new Classification.DataSet.DataSet(dataDefinition, ",", "../../../datasets/car.data");

            attributeTypes = new List<AttributeType>();
            for (var i = 0; i < 9; i++)
            {
                attributeTypes.Add(AttributeType.DISCRETE);
            }

            dataDefinition = new DataDefinition(attributeTypes);
            tictactoe = new Classification.DataSet.DataSet(dataDefinition, ",", "../../../datasets/tictactoe.data");
            attributeTypes = new List<AttributeType>();
            attributeValueList = new string[9][];
            for (int i = 0; i < 9; i++){
                attributeTypes.Add(AttributeType.DISCRETE_INDEXED);
                attributeValueList[i] = new []{"b", "o", "x"};
            }
            dataDefinition = new DataDefinition(attributeTypes, attributeValueList);
            tictactoeIndexed = new Classification.DataSet.DataSet(dataDefinition, ",", "../../../datasets/tictactoe.data");

            attributeTypes = new List<AttributeType>();
            for (var i = 0; i < 8; i++)
            {
                attributeTypes.Add(AttributeType.DISCRETE);
            }

            dataDefinition = new DataDefinition(attributeTypes);
            nursery = new Classification.DataSet.DataSet(dataDefinition, ",", "../../../datasets/nursery.data");
            attributeTypes = new List<AttributeType>();
            for (var i = 0; i < 6; i++)
            {
                if (i % 2 == 0)
                {
                    attributeTypes.Add(AttributeType.DISCRETE);
                }
                else
                {
                    attributeTypes.Add(AttributeType.CONTINUOUS);
                }
            }

            dataDefinition = new DataDefinition(attributeTypes);
            chess = new Classification.DataSet.DataSet(dataDefinition, ",", "../../../datasets/chess.data");
        }
    }
}