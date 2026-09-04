from pymongo import MongoClient
from pymongo.database import Database

from app.core.configuration import settings


class DatabaseConnection:
   
    def __init__(self) -> None:
        self.client = MongoClient(
            settings.MONGODB_URL,
            serverSelectionTimeoutMS=5000,
        )

        self.database: Database = self.client[
            settings.MONGODB_DATABASE
        ]

    def check_connection(self) -> bool:
       
        try:
            self.client.admin.command("ping")
            return True
        except Exception:
            return False

    def get_database(self) -> Database:
        
        return self.database

    def close(self) -> None:
       

        self.client.close()


database_connection = DatabaseConnection()


def get_database() -> Database:
   

    return database_connection.get_database()