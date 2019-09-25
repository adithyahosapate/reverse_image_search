import elasticsearch
def main():
    es = elasticsearch.Elasticsearch()
    print(es.search(index="clustered-images", body={"query": {"match_all": {}}}))
    # es.delete(index='clustered-images', id='kmEjaG0Bg1Dafb9haLex')
if __name__ == '__main__':
    main()
