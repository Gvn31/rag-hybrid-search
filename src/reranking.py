import numpy as np

from sentence_transformers import CrossEncoder

from src.config import RERANKER_MODEL, RERANK_TOP_K
from src.logger import logger
from src.exception import RAGPipelineException

#Store the cross encoder model
_reranker_model=None


#Load the cross encoder model only when required
def get_reranker_model():
    global _reranker_model
    
    try:
        
        if _reranker_model is None:
           logger.info(f"Loading the Re-Ranker model: {RERANKER_MODEL}")
           
           _reranker_model = CrossEncoder(RERANKER_MODEL,max_length=512)
           logger.info("Cross-Encode Model loaded successfully")

        return _reranker_model
        
    
    except Exception as e:
        logger.exception("Failed to load the Re-Ranker model")
        raise RAGPipelineException(str(e))
    
    #Validate the input query and retrieved document chunks
    
def validate_input(query, retrieved_chunks, top_k):
        
    try:
            
        if not isinstance(query, str) or not query.strip():
                raise ValueError("Query should be a non-empty string")
            
        if not isinstance(retrieved_chunks, list):
                raise ValueError("Retrieved chunks should be a list")
            
        if not isinstance(top_k,int) or top_k <= 0:
                raise ValueError("Top_k should be a positive integer")
            
        #Validate individual retrieved chunks
        for chunk in retrieved_chunks:
                if not isinstance(chunk, dict):
                    raise ValueError("Each retrieved chunk should be a dictionary")
                
                if "text" not in chunk or not isinstance(chunk["text"], str) or not chunk["text"].strip():
                    raise ValueError("Each chunk must contain valid text")
                
                if not chunk["text"].strip():
                    raise ValueError("Retrieved chunk text cannot be empty")
            
        return True
        
    except Exception as e:
            logger.exception("Reranking input validation failed")
            raise RAGPipelineException(str(e))
        
    
        
#Calculate relevance scores for query-document pairs
def calculate_relevance_scores(query,retrieved_chunks):
        
    try:
            if not retrieved_chunks:
                return[]
            
            #Load Cross-Encoder Model
            model=get_reranker_model()
            
            #Create query-document pairs of cross-encoder evaluation
            setence_pairs=[
                (query,chunk["text"]) for chunk in retrieved_chunks
            ]           
            
            #Predict relevance scores
            scores=model.predict(
                setence_pairs, batch_size=16, show_progress_bar=False
                )
            
            #Convert model output to a consistent list
            scores=np.asarray(scores).reshape(-1)
            
            if len(scores) != len(retrieved_chunks):
                raise ValueError(
                    "Relevance does not match chunk counts"
                )
            
            #Validate generated scores
            if not np.all(np.isfinite(scores)):
                raise ValueError("Cross-Encoder generated invalid scores")
            
            logger.info(f"Calculated relevance scores for {len(scores)} chunks")
            
            return scores.tolist()
        
    except RAGPipelineException:
            raise
    except Exception as e:
            logger.exception("Failed to calculate relevance scores")
            raise RAGPipelineException(str(e))
        
        
#Rerank retrieved chunks using cross-encoder relevance scores
def rerank_results(
        query, retrieved_chunks,top_k=RERANK_TOP_K
    ):
        try:
            #Validate input
            validate_input(query,retrieved_chunks,top_k)
           
            if not retrieved_chunks:
                logger.warning("No retrieved chunks available for reranking")
                return []
            
            #Calculate relevance scores for retrieved chunks
            relevance_scores = calculate_relevance_scores(query, retrieved_chunks)
            
            #Attach reranking scores to the original chunks
            reranked_chunks = []
            
            for chunk, score in zip(retrieved_chunks, relevance_scores):
                
                #Preserve the original chunk info and metadata
                reranked_chunk=chunk.copy()
                reranked_chunk["rerank_score"] =float(score)
                reranked_chunks.append(reranked_chunk)
                
            #Sort the chunks by relevance score in descending order
            reranked_chunks.sort(
                key=lambda chunk: chunk["rerank_score"], reverse=True
                )
               
            #Select the final top_ranked chunks
            final_results=reranked_chunks[:top_k]
            logger.info(f"Reranking completed: {len(final_results)} final chunks selected")
            return final_results
        
        except RAGPipelineException:
            raise
        
        except Exception as e:
            logger.exception("Cross-Encoder Reranking failed")
            raise RAGPipelineException(str(e))
            
    
#Execute hybrid retrieval followed by cross-encoder reranking
def retrieve_and_rerank(
    query,
    top_k=RERANK_TOP_K,
    retrieval_top_k=10,
    user_id=None,
    where=None
):
    try:
        from src.retrieval import hybrid_retrieval

        # Validate the requested result count
        if not isinstance(retrieval_top_k, int) or retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be a positive integer")

        logger.info("Starting retrieval and reranking pipeline")

        # Step 1: Retrieve candidate chunks using hybrid search
        retrieved_chunks = hybrid_retrieval(
            query=query,
            top_k=retrieval_top_k,
            user_id=user_id,
            where=where
        )

        # Step 2: Rerank candidates using the cross-encoder
        final_results = rerank_results(
            query=query,
            retrieved_chunks=retrieved_chunks,
            top_k=top_k
        )

        logger.info(
            f"Retrieval and reranking completed with "
            f"{len(final_results)} results"
        )

        return final_results

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Retrieval and reranking pipeline failed")
        raise RAGPipelineException(str(e))


#Test cross-encoder reranking independently
def main():
    try:
        # Create sample hybrid retrieval results
        sample_results = [
            {
                "id": "chunk_001",
                "text": (
                    "Artificial intelligence enables machines "
                    "to perform tasks requiring human intelligence."
                ),
                "metadata": {
                    "source": "reranking_test.txt",
                    "chunk_id": "chunk_001"
                },
                "rrf_score": 0.025,
                "dense_rank": 2,
                "sparse_rank": 3
            },
            {
                "id": "chunk_002",
                "text": (
                    "Machine learning allows computers to learn "
                    "patterns from data."
                ),
                "metadata": {
                    "source": "reranking_test.txt",
                    "chunk_id": "chunk_002"
                },
                "rrf_score": 0.028,
                "dense_rank": 1,
                "sparse_rank": 1
            },
            {
                "id": "chunk_003",
                "text": (
                    "Deep learning uses neural networks "
                    "to understand complex data."
                ),
                "metadata": {
                    "source": "reranking_test.txt",
                    "chunk_id": "chunk_003"
                },
                "rrf_score": 0.024,
                "dense_rank": 3,
                "sparse_rank": 2
            }
        ]

        # Define the test query
        query = "How do computers learn from data?"

        print(f"\nSearch query: {query}")

        # Execute cross-encoder reranking
        reranked_results = rerank_results(
            query=query,
            retrieved_chunks=sample_results,
            top_k=3
        )

        # Display reranked results
        print("\nCross-Encoder Reranking Results:")

        for rank, result in enumerate(reranked_results, start=1):

            print(f"\nRank: {rank}")
            print("Chunk ID:", result["id"])
            print("Text:", result["text"])
            print("RRF Score:", result["rrf_score"])
            print("Reranking Score:", result["rerank_score"])
            print("Metadata:", result["metadata"])

        print(
            f"\nTotal reranked chunks: {len(reranked_results)}"
        )

    except RAGPipelineException as e:
        logger.error(f"Reranking test failed: {e}")
        print(f"Reranking test failed: {e}")

    except Exception as e:
        logger.exception("Unexpected error in reranking module")
        print(f"Unexpected error: {e}")


# Run the test when this file is executed directly
if __name__ == "__main__":
    main()
