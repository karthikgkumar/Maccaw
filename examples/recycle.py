import os
import threading
import winsound
from pydantic import BaseModel, Field
from typing import Optional, Type
from maccaw.maccaw import BaseTool, OpenaiLLM
from maccaw.main import MaccawAgent
import win32com.client
import winshell
from datetime import datetime, timedelta
from winshell import ShellRecycledItem

llm = OpenaiLLM(
    base_url=os.getenv('BASE_URL'),
    api_key=os.getenv('API_KEY'),
    model=os.getenv("MODEL_NAME"),
    max_tokens=120,
)

class RecycleBinArgs(BaseModel):
    action: str = Field(
        description="The action to perform: 'clear_bin', 'recycle_file', 'search_file'  or 'recycle_all'.",
        enumerate=['clear_bin', 'recycle_file','recycle_all', 'search_file']
    )
    file_name:str = Field(
        description="The name of the file/folder to recycle (required for 'recycle_file' action)."
    )
    file_path: str= Field(
        description="The path of the file/folder to recycle.",default=None
    )

class RecycleBinTool(BaseTool):
    name: str = "recycle_bin"
    description: str = "Clear the recycle bin or recycle a specific file or all files."
    args_schema: Optional[Type[BaseModel]] = RecycleBinArgs
    return_direct: Optional[bool] = False

    def _run(self, action: str, file_name:Optional[str] = None , file_path: Optional[str] = None) -> str:
        from winshell import ShellRecycledItem
        import win32con
        import winshell
        import os

        if action=='search_file':
            if file_name is None:
                return "Error: file_path is required for 'recycle_file' action."
            try:
                item=winshell.recycle_bin()
                r = list(winshell.recycle_bin())  # this lists the original path of all the all items in the recycling bin\
                if len(r)!=0:
                    similar_filenames = []
                    for index,items in enumerate(r):
                        print(index)
                        filepath=ShellRecycledItem.original_filename(items)
                        # winshell.undelete(filepath)    
                        
                        filename= os.path.basename(filepath)
                        similar_filenames.append({"file_name": filename, "file_path": filepath})
                        print("similar file", similar_filenames)
                    prompt = f"""Below are the items in {similar_filenames}. Assign suitable 'file_path' to 'RecycleBinTool'"""
                    return prompt
                else:
                    return f"Error: '{file_name}' is not in recycle bin"
            except Exception as e:
                return f"Error occurred while recycling the file: {str(e)}"
        if action == "clear_bin":
            try:
                recycle_bin = winshell.recycle_bin()
                items=len(list(winshell.recycle_bin()))
                if items!=0:
                    recycle_bin.empty(confirm=True, show_progress=True, sound=True)
                    return "Recycle bin has been cleared."
                else:
                    return "The recycle bin is already empty."
            except Exception as e:
                return f"Error occurred while clearing the recycle bin: {str(e)}"
        
        
        elif action == "recycle_file":
            if file_name is None:
                return "Error: file_path is required for 'recycle_file' action."
            try:
                
                item=winshell.recycle_bin()
                r = list(winshell.recycle_bin())  # this lists the original path of all the all items in the recycling bin\
                if len(r)!=0:
                    winshell.undelete(file_path) 
                    return f"the file {file_name} has been restored"
                else:
                    return f"Error: '{file_name}' is not in recycle bin"
            except Exception as e:
                return f"Error occurred while recycling the file: {str(e)}"
        
        elif action == "recycle_all":
            try:
                recycle_bin = winshell.recycle_bin()
                r=list(recycle_bin)
                if r!=0:
                    for index,item in enumerate(r):
                        print(index)
                        filepath=ShellRecycledItem.original_filename(item)
                        winshell.undelete(filepath)                  
                    return "All files have been recovered from recycle bin."
                else:
                    return "The recycle bin is already empty."
            except Exception as e:
                return f"Error occurred while recycling all files: {str(e)}"
        else:
            return "Error: Invalid action specified."

sys_prompt = """
You are a helpful assistant that can clear the recycle bin on the local system, recycle a specific file, or recycle all files. The user may ask you to perform any of these actions.
For recycling specific files, use 'search_file' to search for the files and retrieve filepath. Then recycle the files.
If there are any issues while performing these actions, you will provide an appropriate error message.
"""


tools = [RecycleBinTool()]

agent = MaccawAgent(
    sys_prompt=sys_prompt,
    maccaw_llm=llm,
    tools_list=tools,
    logging=True,
    use_system_prompt_as_context=True,
    pickup_mes_count=6
)

while True:
    user_input = input("Human: ")
    if user_input.lower() == "exit":
        break
    response = agent.run(user_input)
    print("Agent:", response)
