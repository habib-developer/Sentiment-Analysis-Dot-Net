using Microsoft.ML;
using SentimentAnalysisDotNetML.Model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;

namespace SentimentAnalysisDotNet.Controllers
{
    public class SemanticController : Controller
    {
        [HttpGet]
        public ActionResult Analysis()
        {
            return View();
        }
        [HttpPost]
        public ActionResult Analysis(ModelInput input)
        {
            // Load the model  
            MLContext mlContext = new MLContext();
            var path = Server.MapPath(@"~\SentimentAnalysisDotNetML.Model\MLModel.zip");
            ITransformer mlModel = mlContext.Model.Load(path, out var modelInputSchema);
            // Create predection engine related to the loaded train model
            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);
            //Input  
            input.Year = DateTime.Now.Year;
            // Try model on sample data and find the score
            ModelOutput result = predEngine.Predict(input);
            // Store result into ViewBag
            ViewBag.Result = result;
            return View();
        }
    }
}