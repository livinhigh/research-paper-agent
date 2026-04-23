#Answer by Ivan Joseph ( M01093025 )

Dear grandma , 

this is application is a website that is run in browsers , which helps computers to connect to different website in the Internet.
The website is called Research Paper Agent which uses Artificial Intellgience ( a very revolutionary discovery in the recent years which mimic human intelligence).
When you open the website , the website will ask you to upload any research paper in PDF format, which is a widely used document format type, and you can ask questions related to the paper or PDF document that is uploaded , along with a facility to search the web for more information needed.
After the PDF is uploaded , the contents in the PDF is split into chunks , using text-embedding models like all-MiniLM-L6-v2 and sentence-transformers, similar meaning sentences are put together into a vector DB with certain index , vector DB used is Chroma DB

This is done so that when the application gives back response to the user , it has proper context , page number and ensure there is no data hallucination.

There is a guardrail layer in this application , where it checks if your questions/answers generated are related to the PDF uploaded or not, the application only allows questions/answers related to the PDF , and blocks any other type of content.

Generative AI is a major part of the application where we use models like llama-3.3-70b to generate the text content , to decide the agent to use etc.

There are multiple agents in this application :
Router Node 
RAG Agent
Web Agent
Summarizer
Synthesizer Node

Router Node is the first step which will decide which Agent to use among RAG , Web , Summarizer.
After the Agents does it job , the last step is Synthesizer Node which compiles all response of all agents and shows the content back in text format for user to see.

This application will help students/researchers/teachers etc to quickly learn what is in the document and will be a great tool in the academic sector.

#Answer by Ihsan Ul Haque ( M01098089 )