import elasticsearch


class Setup:
    def __init__(self, host):
        self.es = elasticsearch.Elasticsearch([{'host': host}])
        self.mapping = '''
                {  
                  "mappings":{  
                    "properties": {
                        "imageId": { 
                            "type": "keyword",
                            "index": false 
                        },  
                        "clusterString":  {
                            "type": "text",
                            "index_prefixes": {
                                "min_chars" : 8,
                                "max_chars" : 8
                            }
                        }
                    }
                  }
                }'''

    def init_es_cluster(self):
        print(self.es.indices.create(index='clustered-images', ignore=400, body=self.mapping))


if __name__ == '__main__':
    setup = Setup('localhost')
    setup.init_es_cluster()
