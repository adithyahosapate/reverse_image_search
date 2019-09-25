import elasticsearch
import elasticsearch.helpers

class EsDao:
    clustered_images = 'clustered-images'

    def __init__(self, host):
        self.es = elasticsearch.Elasticsearch([{'host': host}])

    def store_clustered_image(self, image_id, cluster_string):
        document = {"imageId": image_id, "clusterString": cluster_string}
        if self.es.index(index=self.clustered_images, body=document)['result'] == 'created':
            return True
        else:
            return False

    def get_image(self, id):
        return self.es.get(index=self.clustered_images, id=id)

    def store_in_bulk(self, image_cluster_mapping):
        actions = []
        for image_id in image_cluster_mapping:
            actions.append({
                "_index": self.clustered_images,
                "_type": "_doc",
                "_source": {
                    "imageId": image_id,
                    "clusterString": image_cluster_mapping[image_id]
                }
            })
        return elasticsearch.helpers.bulk(self.es, actions)
