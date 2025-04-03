
#[macro_export]
macro_rules! error_to_response {
    ($res:ident) => {
        match $res {

            Err(infer::AppError::Message(s)) => {
                println!("{}", s);
            }
            _ => {
                println!("Unknown error");
            }
        }
    };
}
