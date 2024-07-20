import scrapy
import json
import argparse
import logging
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from scrapy.http import Request
from PIL import Image
from io import BytesIO
from google.cloud import storage

class ImageSpider(scrapy.Spider):
    name = 'image_spider'

    def __init__(self, config_file):
        super().__init__()
        self.config = self.load_config(config_file)
        self.base_url = 'https://example.com/'  # Replace with actual base URL
        self.logger = logging.getLogger('ImageSpider')

    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            return json.load(f)

    def start_requests(self):
        for animal in self.config['animals']:
            url = f"{self.base_url}/{animal}"
            yield Request(url, callback=self.parse, meta={'animal': animal})

    def parse(self, response):
        animal = response.meta['animal']
        image_urls = response.css('img::attr(src)').getall()
        image_urls = [response.urljoin(url) for url in image_urls]
        image_urls = image_urls[:self.config['images_per_category']]

        for img_url in image_urls:
            yield Request(img_url, callback=self.save_image,
                          meta={'animal': animal})

    def save_image(self, response):
        try:
            img = Image.open(BytesIO(response.body))
            img_width, img_height = img.size
            min_width = self.config['minimum_size']['width']
            min_height = self.config['minimum_size']['height']

            if img_width >= min_width and img_height >= min_height:
                bucket_name = self.config['gcs_bucket']
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(f"{response.meta['animal']}/{response.url.split('/')[-1]}")
                blob.upload_from_string(response.body)
                self.logger.info(f"Saved {response.url} to GCS bucket {bucket_name}")
            else:
                self.logger.warning(f"Ignored {response.url} due to insufficient size")
        except Exception as e:
            self.logger.error(f"Failed to process image {response.url}: {str(e)}")

def main(config_file):
    process = CrawlerProcess(get_project_settings())
    process.crawl(ImageSpider, config_file=config_file)
    process.start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scrapy script to scrape images and store them in GCS')
    parser.add_argument('config', help='Path to JSON configuration file')
    args = parser.parse_args()

    main(args.config)
