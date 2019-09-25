import es_index_image

if __name__ == '__main__':
    es = es_index_image.ClusterToWord()
    es.create_mapping([i for i in range(5000)])
    es.dump_mapping()

