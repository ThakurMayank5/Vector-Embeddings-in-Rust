use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};
use std::time::Instant;

fn main() {
    let start_total = Instant::now();

    let start_load = Instant::now();
    let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
        .create_model()
        .unwrap();
    let load_duration = start_load.elapsed();
    println!("Model loaded in {:.2?}", load_duration);

    let sentences = [
        "This is a document about oranges",
        "This is a query document about hawaii",
    ];

    let start_embed = Instant::now();
    let output = model.encode(&sentences).unwrap();
    let embed_duration = start_embed.elapsed();
    println!("Embeddings generated in {:.2?}", embed_duration);

    println!("Output embeddings: {:?}", output);

    let total_duration = start_total.elapsed();
    println!("Total time taken: {:.2?}", total_duration);
}
