// const express = require("express");
// const bodyParser = require("body-parser");
// const app = express();
// const port = 3000;

// //load pre-trained model

// const { CountVectorizer, loadModel } = require("scikit-learn");
// const model = loadModel("/model/train-test/KNN_train_test.pkl");
// // const { load } = require("picklejs");

// // //load pre-trained model
// // const model = load("/model/train-test/RNN_train_test.pkl");

// // const vectorizer = new CountVectorizer();

// // Set the view engine to ejs
// app.set("view engine", "ejs");
// app.use(express.static("public"));

// // Parse URL-encoded bodies
// app.use(bodyParser.urlencoded({ extended: false }));

// // Parse JSON bodies
// app.use(bodyParser.json());

// app.get("/", (req, res) => {
//   res.render("prediction");
//   // Perform prediction using the input data
//   // const prediction = performPrediction(inputData);

//   // res.json({ prediction });
// });

// app.post("/submit", (req, res) => {
//   const formData = req.body;

//   loadModal(formData);
//   // Process the form data here or send a response back
//   // res.send("Received form data");
// });

// app.listen(port, () => {
//   console.log(`App listening at http://localhost:${port}`);
// });

// function loadModal(formData) {
//   console.log(formData);
//   formData = JSON.stringify(formData);
//   console.log(formData);
//   if (formData.split == "KFold" && formData.model == "KNN") {
//     console.log("KNN KFold");
//   } else {
//     console.log("KNN Train Test");
//   }
// }