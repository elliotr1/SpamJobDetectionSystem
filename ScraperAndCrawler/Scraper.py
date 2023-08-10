import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
import mysql.connector

con = mysql.connector.connect(
    database="scrapedJobs",
    host="localhost",
    user="elliot",
    password="Jasmner5555.",
    charset='utf8mb4'
)

curs = con.cursor()


def scrapReedJobs():
    for i in range(341, 8000):
        print(f"https://www.reed.co.uk/jobs?sortBy=displayDate&proximity=50&pageno={i}")
        print(f"Round: {i}")
        content = requests.request("get",
                                   f"https://www.reed.co.uk/jobs?sortBy=displayDate&proximity=50&pageno={i}").content
        soup = BeautifulSoup(content, "html.parser")
        r = soup.find("main")
        job_cards = r.find_all("a", {"data-qa":"job-card-title"}, href=True)
        jobs = {f"https://www.reed.co.uk{job['href']}" for job in job_cards if
                job['href'].startswith("/") and "resourcery-group" not in job['href'] and "/jobs/jobs-in" not in job[
                    "href"] and "jobwise" not in job["href"] and not re.match(r'^p\d{6}', job["href"])}
        jobs = [job for job in jobs]
        for job in jobs:
            content = requests.request("get", job).content
            soup = BeautifulSoup(content, "html.parser")
            result = soup.find("span", {"itemprop": "description"})
            if result is not None:
                company_name = soup.find("span", {"data-page-component": "job_description"})
                if company_name is not None:
                    company_name = company_name.text
                paragraphs = result.find_all("p")
                description = ""
                company_profile = ""
                strong_encountered = False

                for paragraph in paragraphs:
                    if not strong_encountered:
                        if paragraph.find("strong"):
                            strong_encountered = True
                        description += " " + paragraph.text.strip()
                    else:
                        company_profile += " " + paragraph.text.strip()
                if len(description) == 0:
                    continue
                title = soup.find("h1").text
                salary = soup.find("span", {"data-qa": "salaryLbl"})
                if salary is not None:
                    salary = salary.text
                required_skills = ""
                skills_required = soup.find("ul", {"class": "list-unstyled skills-list"})
                if skills_required is not None:
                    skills_required = skills_required.find_all("li")
                    for skill in skills_required:
                        required_skills += skill.text + " "
                area = soup.find("span", {"data-qa": "regionLbl"})
                county = soup.find("span", {"data-qa": "localityLbl"})
                if area is not None and county is not None:
                    area = area.text + ", " + county.text

                requirements = result.find_all("li")
                requirements_str = ""
                for requirement in requirements:
                    requirements_str += " " + requirement.text
                requirements_str.strip()
                values = {
                    "title": title,
                    "description": description,
                    "requirements_str": requirements_str,
                    "company_name": company_name,
                    "salary": salary
                }
                encoded_values = {}
                for key, value in values.items():
                    try:
                        encoded_values[key] = value.encode('utf-8')
                    except Exception as e:
                        print(f"Error encoding {key}: {e}")
                        encoded_values[key] = ""

                if all((value is None or isinstance(value, bytes)) for value in encoded_values.values()):
                    values_tuple = (title, description, requirements_str, company_profile, salary)
                    curs.execute(
                        "SELECT COUNT(*) FROM Jobs WHERE title=%s AND description=%s AND requirements=%s AND company_profile=%s AND salary=%s",
                        values_tuple)
                    count_result = curs.fetchone()  # Fetch the result
                    row_count = count_result[0]  # Access the count value

                    if row_count == 0:
                        # Perform the insertion
                        query = "INSERT INTO jobs(title, description, requirements, company_profile, salary) VALUES (%s, %s, %s, %s, %s)"
                        curs.execute(query, values_tuple)
                        con.commit()
                        print("Successfully inserted values into db.")
                    else:
                        print("Value is already in the database")
