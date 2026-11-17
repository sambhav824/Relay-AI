import os
import sys
from PySide6.QtWidgets import QApplication
from src.context.auth.manager import AuthManager
from src.context.db.connection import initialize_database
from src.context.documents.manager import DocumentManager
from src.context.repository import ContextRepository
from src.context.retrieval.hybrid import HybridRetriever
from src.context.retrieval.sparse_client import SparseRetrievalClient
from src.context.retrieval.vector_client import VectorSimilarityClient
from src.context.sync.queue import SyncQueueManager
from src.context.sync.worker import SyncWorker
from src.ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Relay AI")
    app.setOrganizationName("Relay AI")

    # Initialize local context DB and sync outbox
    db = initialize_database()
    conn = db.get_connection()
    sync_queue = SyncQueueManager(conn)
    auth_manager = AuthManager()
    try:
        auth_manager.restore_session()
    except Exception:
        pass

    current_user_id = auth_manager.user.id if auth_manager.user else "local"
    context_repo = ContextRepository(conn, sync_queue=sync_queue, default_user_id=current_user_id)

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_anon_key = os.environ.get("SUPABASE_ANON_KEY")

    sync_worker = None
    if supabase_url and supabase_anon_key and not supabase_url.startswith("placeholder"):
        sync_worker = SyncWorker(
            queue=sync_queue,
            auth_manager=auth_manager,
            supabase_url=supabase_url,
            supabase_anon_key=supabase_anon_key,
        )
        sync_worker.start()

    # Initialize document manager and hybrid RAG retrieval
    doc_manager = DocumentManager(
        connection=conn,
        user_id=current_user_id,
        sync_queue=sync_queue,
    )

    sparse_client = SparseRetrievalClient(
        supabase_url=supabase_url,
        supabase_anon_key=supabase_anon_key,
        auth_manager=auth_manager,
        sqlite_connection=conn,
    )
    vector_client = None
    if supabase_url and supabase_anon_key and not supabase_url.startswith("placeholder"):
        vector_client = VectorSimilarityClient(
            supabase_url=supabase_url,
            supabase_anon_key=supabase_anon_key,
            auth_manager=auth_manager,
        )

    hybrid_retriever = HybridRetriever(
        vector_client=vector_client,
        sparse_client=sparse_client,
    )

    window = MainWindow(
        sync_worker=sync_worker,
        auth_manager=auth_manager,
        sync_queue=sync_queue,
        context_repo=context_repo,
        document_manager=doc_manager,
        hybrid_retriever=hybrid_retriever,
    )
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
