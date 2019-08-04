import click
import requests

autotrader_url = 'http://myslu.stlawu.edu/~clee/dataset/autotrader/retrieve.php?'

@click.command()
@click.option('--car_make', prompt='Car make', help='The brand of the car')
@click.option('--car_model', prompt='Car model', help='The model of the car')
@click.option('--zip_code', prompt='Zip code', help='Your zip code')
@click.option('--radius', default=100, help='Radius of car searches with respect to zip code')
@click.option('--search_results', default=300, help='Amount of search results')
def get_user_car_info(car_make, car_model, zip_code, radius, search_results):
    click.echo('Grabbing data from Autotrader.com for %s %s at location %s' % (car_make, car_model, zip_code))
    tuple_of_car_info = (car_make, car_model, zip_code, radius, search_results)
    get_autotrader_data(tuple_of_car_info)

def get_autotrader_data(tuple_of_car_info):
    temp_autotrader_url = autotrader_url
    temp_autotrader_url += ('make=%s' % tuple_of_car_info[0].upper())
    temp_autotrader_url += ('&model=%s' % tuple_of_car_info[1].upper())
    temp_autotrader_url += ('&zipcode=%s' % tuple_of_car_info[2])
    temp_autotrader_url += ('&radius=%s' % tuple_of_car_info[3])
    temp_autotrader_url += ('&limit=%s' % tuple_of_car_info[4])
    resp = requests.get(temp_autotrader_url)
    data_text = resp.text
    with open('data.csv', 'w') as f:
        f.write(data_text)


if __name__ == '__main__':
    get_user_car_info()