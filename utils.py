# TODO: abstract all of this into a function that takes in a PDF file name

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from typing import List, Optional
from llama_index.core import (
    load_index_from_storage,
    StorageContext,
)

# def get_doc_tools(
#     file_path: str,
#     name: str,
# ) -> str:
#     """Get vector query and summary query tools from a document."""

#     # load documents
#     documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
#     splitter = SentenceSplitter(chunk_size=1024)
#     nodes = splitter.get_nodes_from_documents(documents)
#     vector_index = VectorStoreIndex(nodes)

#     def vector_query(
#         query: str,
#         filter_key_list: List[str],
#         filter_value_list: List[str]
#     ) -> str:
#         """Perform a vector search over an index.

#         query (str): the string query to be embedded.
#         filter_key_list (List[str]): A list of metadata filter field names
#             Must specify ['page_label'] or empty list. Please leave empty
#             if there are no explicit filters to specify.
#         filter_value_list (List[str]): List of metadata filter field values
#             (corresponding to names specified in filter_key_list)

#         """
#         metadata_dicts = [
#             {"key": k, "value": v} for k, v in zip(filter_key_list, filter_value_list)
#         ]

#         query_engine = vector_index.as_query_engine(
#             similarity_top_k=2,
#             filters=MetadataFilters.from_dicts(metadata_dicts)
#         )
#         response = query_engine.query(query)
#         return response

#     vector_query_tool = FunctionTool.from_defaults(
#         fn=vector_query,
#         name=f"vector_query_{name}"
#     )

#     summary_index = SummaryIndex(nodes)
#     summary_query_engine = summary_index.as_query_engine(
#         response_mode="tree_summarize",
#         use_async=True,
#     )
#     summary_tool = QueryEngineTool.from_defaults(
#         query_engine=summary_query_engine,
#         name=f"summary_query_{name}",
#         description=(
#             f"Useful for summarization questions related to {name}"
#         ),
#     )
#     return vector_query_tool, summary_tool



def get_doc_tools(
    file_path: str,
    name: str,
    Settings,
) -> str:
    """Get vector query and summary query tools from a document."""


    try:
        # rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir=f"./{name}")
        # load index
        vector_index = load_index_from_storage(storage_context, index_id=f"{name}")
        print('loading index from memory')
    except:
        print('creating and saving index to memory')
        # load documents
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        splitter = SentenceSplitter(chunk_size=1024)
        nodes = splitter.get_nodes_from_documents(documents,show_progress =True)
        vector_index = VectorStoreIndex(nodes, embed_model = Settings.embed_model)
        # save index to disk
        vector_index.set_index_id(f"{name}")
        vector_index.storage_context.persist(f"./{name}")

    def vector_query(
        query: str
    ) -> List[str]:
        """Use to answer questions over a given paper.

        Useful if you have specific questions over the paper.
        Always leave page_numbers as None UNLESS there is a specific page you want to search for.

        Args:
            query (str): the string query to be embedded.
            page_numbers (Optional[List[str]]): Filter by set of pages. Leave as NONE
                if we want to perform a vector search
                over all pages. Otherwise, filter by the set of specified pages.

        """

        #query_engine = vector_index.as_query_engine(
        #    similarity_top_k=2,
        #    llm = groq_llm
        #)
        query_engine = vector_index.as_retriever(similarity_top_k = 1)
        response = query_engine.retrieve(query)
        print('-----------------------------------------')
        print(len(response))
        first_node = response[0]
        return f"according to page {first_node.metadata['page_label']} in {first_node.metadata['file_name']} : {first_node.get_text()}"
        #return [(node.get_text(),
        #        f"page number:  {node.metadata['page_label']}",
        #        f" SOURCE:  {node.metadata['file_name']}") for node in response
        #        ]
    


    vector_query_tool = FunctionTool.from_defaults(
        name=f"vector_tool_{name}",
        fn=vector_query,
        description= f"this function addresses and searches a vector database for all queries regarding {name}"
    )


    #summary_index = SummaryIndex(nodes, show_progress  =True)

    #summary_query_engine = summary_index.as_query_engine(
    #    response_mode="tree_summarize",
    #    use_async=True,
    #    llm = groq_llm
    #)
    #summary_tool = QueryEngineTool.from_defaults(
    #    name=f"summary_tool_{name}",
    #    query_engine=summary_query_engine,
    #    description=(
    #        f"Useful for summarization questions related to {name}"
    #    ),
    #)


    return vector_query_tool#, summary_tool