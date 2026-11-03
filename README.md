# RetinaView

Streamlit prototype for retinal image classification and test history.

## Deploy
1. Upload these files to a GitHub repository.
2. Open https://share.streamlit.io and create an app using `app.py`.
3. In Streamlit app settings, add the following secrets (never commit these to GitHub):

```toml
SUPABASE_URL = "your Supabase project URL"
SUPABASE_SERVICE_ROLE_KEY = "your Supabase service-role key"
```

The Supabase project must already contain the `patients` and `tests` tables and the `fundus-images` storage bucket expected by the app.

The model is downloaded from Hugging Face when first needed. First startup may take a while.
