```python


client_id = ""
client_secret = ""
client_scopes = ""

os.environ["REQUESTS_CA_BUNDLE"] = r"utils/CitiInternalCAChain_PROD.pem"
os.environ["USERNAME"] = 'sk92484'

def get_api_key():

    url = 'https://coin-uat.ls.dyn.nsroot.net/token/v2/' + client_id
    header = {
        'accept': '*/*',
        'Content-Type': 'application/json'
    }

    payload = {
        'clientSecret': client_secret,
        'clientScopes': client_scopes
    }

    api_key = requests.post(url, json=payload, headers=header, verify=False)
    token = api_key.text
    return token


class GenAI_UDC:
    def __init__(self,modelname="gemini-2.5-flash",genConfig_file=None):
        credentials = Credentials(token=get_api_key())
        vertexai.init(
            project="prj-gen-ai-9571",  # uses  UAT PROJECT
            api_transport="rest",
            api_endpoint="https://r2d2-c3p0-icg-msst-genaihub-178909.apps.namicg39023u.ecs.dyn.nsroot.net/vertex",  # uses R2D2 UAT
            credentials=credentials,
            request_metadata=[("x-r2d2-user", os.getenv("USERNAME"))],
        )

        self.gen_model = GenerativeModel(modelname)
        self.genConfig_file = genConfig_file
        if genConfig_file:
            self.genConfig_schema = self.get_LLMConfig(genConfig_file)
        else:
            self.genConfig_schema = None


    def get_LLMConfig(self,genConfig_file):
        with open(genConfig_file, 'r') as f:
            return json.load(f)


    def getContent(self,prompt):
        genConfig = {}
        if self.genConfig_schema:
            genConfig = GenerationConfig(
                response_mime_type = "application/json",
                response_schema = self.genConfig_schema
            )

        resp = self.gen_model.generate_content(
            prompt,
            generation_config = genConfig
        )
        return resp